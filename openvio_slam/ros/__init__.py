self._pose_cache = {}   # frame_id -> (R_GC, p_GC)
self._initial_p_GI = np.zeros(3)
self._initial_q_GI = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
self._frame_times = []
self._n_imu = 0
self._last_imu_time = None