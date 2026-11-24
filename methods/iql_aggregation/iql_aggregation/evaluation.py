from typing import Any, Dict, Optional

import flax.linen as nn
import gym
import numpy as np
import os
import imageio


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int,
             record_video: bool = False,
             video_dir: Optional[str] = None,
             video_prefix: str = "eval",
             video_fps: int = 30,
             render_width: int = 640,
             render_height: int = 480,
             camera_name: Optional[str] = "track",
             follow_agent_camera: bool = False,
             follow_camera_distance: float = 18.0,
             follow_camera_elevation: float = -90.0,
    follow_camera_azimuth: float = 90.0) -> Dict[str, Any]:
    stats = {'return': [], 'length': []}

    def _extract_image_like_frame(frame):
        if frame is None:
            return None
        if isinstance(frame, dict):
            for key in ("rgb", "image", "frame", "rgb_array"):
                if key in frame:
                    return _extract_image_like_frame(frame[key])
            for value in frame.values():
                candidate = _extract_image_like_frame(value)
                if candidate is not None:
                    return candidate
            return None
        if isinstance(frame, (list, tuple)):
            for item in frame:
                candidate = _extract_image_like_frame(item)
                if candidate is not None:
                    return candidate
            return None
        return frame

    def _to_hwc_uint8(frame):
        img = _extract_image_like_frame(frame)
        if img is None:
            return None

        arr = np.asarray(img)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] >= 3:
                arr = arr[..., :3]
            else:
                return None
        else:
            return None

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and np.nanmax(arr) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _prepare_video_frames(raw_frames):
        prepared = []
        for frame in raw_frames:
            arr = _to_hwc_uint8(frame)
            if arr is None:
                continue
            prepared.append(arr)

        if not prepared:
            return []

        min_h = min(f.shape[0] for f in prepared)
        min_w = min(f.shape[1] for f in prepared)
        prepared = [f[:min_h, :min_w, :3] for f in prepared]
        return prepared

    if record_video:
        if not video_dir:
            raise ValueError("`video_dir` must be provided when `record_video=True`.")
        os.makedirs(video_dir, exist_ok=True)

    def _find_base_env(e):
        base = e
        while hasattr(base, 'env'):
            base = base.env
        return base

    base_env = _find_base_env(env)

    def _apply_follow_camera():
        if not follow_agent_camera:
            return
        if not hasattr(base_env, 'sim'):
            return
        if not hasattr(base_env, '_get_viewer'):
            return

        try:
            viewer = base_env._get_viewer('rgb_array')
            cam = viewer.cam
            qpos = base_env.sim.data.qpos
            cam.trackbodyid = -1
            cam.lookat[0] = float(qpos[0])
            cam.lookat[1] = float(qpos[1])
            cam.lookat[2] = 0.0
            cam.distance = float(follow_camera_distance)
            cam.elevation = float(follow_camera_elevation)
            cam.azimuth = float(follow_camera_azimuth)
        except Exception:
            pass

    def _render_rgb_frame():
        _apply_follow_camera()

        kwargs = {"mode": "rgb_array"}
        if render_width is not None:
            kwargs["width"] = render_width
        if render_height is not None:
            kwargs["height"] = render_height
        if camera_name:
            kwargs["camera_name"] = camera_name

        frame = None
        try:
            frame = env.render(**kwargs)
        except TypeError:
            no_cam_kwargs = dict(kwargs)
            no_cam_kwargs.pop("camera_name", None)
            try:
                frame = env.render(**no_cam_kwargs)
            except TypeError:
                frame = env.render(mode="rgb_array")
        except Exception:
            no_cam_kwargs = dict(kwargs)
            no_cam_kwargs.pop("camera_name", None)
            try:
                frame = env.render(**no_cam_kwargs)
            except Exception:
                frame = None

        normalized = _to_hwc_uint8(frame)
        if normalized is None and hasattr(base_env, 'sim'):
            try:
                sim_kwargs = {}
                if render_width is not None:
                    sim_kwargs["width"] = render_width
                if render_height is not None:
                    sim_kwargs["height"] = render_height
                if camera_name:
                    sim_kwargs["camera_name"] = camera_name
                frame = base_env.sim.render(**sim_kwargs)
                normalized = _to_hwc_uint8(frame)
            except TypeError:
                try:
                    sim_kwargs.pop("camera_name", None)
                    frame = base_env.sim.render(**sim_kwargs)
                    normalized = _to_hwc_uint8(frame)
                except Exception:
                    normalized = None
            except Exception:
                normalized = None

        return normalized

    for ep_idx in range(num_episodes):
        observation, done = env.reset(), False
        frames = []

        if record_video:
            frame = _render_rgb_frame()
            frames.append(frame)

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            if record_video:
                frame = _render_rgb_frame()
                frames.append(frame)

        episode_return = float(info['episode']['return'])
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if record_video and frames:
            frames = _prepare_video_frames(frames)
            if not frames:
                continue

            ret_tag = f"ret_{episode_return:.2f}"
            base_name = f"{video_prefix}_ep{ep_idx + 1:03d}_{ret_tag}"
            mp4_path = os.path.join(video_dir, f"{base_name}.mp4")
            try:
                imageio.mimsave(mp4_path, frames, fps=video_fps)
            except ValueError:
                gif_path = os.path.join(video_dir, f"{base_name}.gif")
                try:
                    duration = 1.0 / max(video_fps, 1)
                    with imageio.get_writer(
                        gif_path, mode='I', duration=duration, loop=0
                    ) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                except Exception:
                    pass

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
