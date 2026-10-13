from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mult_agent_architecture_files_exist():
    assert (ROOT / 'openvio_slam' / 'imu' / 'imu_publisher_node.py').exists()
    assert (ROOT / 'web_controller' / 'web_controller_node.py').exists()
    assert (ROOT / 'web_controller' / 'static' / 'index.html').exists()
    assert (ROOT / 'launch' / 'openvio_slam.launch.py').exists()


def test_launch_file_includes_new_nodes():
    launch_text = (ROOT / 'launch' / 'openvio_slam.launch.py').read_text()
    assert 'imu_publisher_node' in launch_text
    assert 'web_controller_node' in launch_text
    assert 'publish_imu' in launch_text or 'rosbridge_websocket' in launch_text


def test_static_tf_and_runtime_notes_are_present():
    launch_text = (ROOT / 'launch' / 'openvio_slam.launch.py').read_text()
    assert 'static_transform_publisher' in launch_text
    # Accept either the combined flag form ('--roll-pitch-yaw R P Y')
    # or the per-axis form ('--roll R --pitch P --yaw Y');
    # both are valid for static_transform_publisher in ROS 2 Jazzy.
    assert (
        '--roll-pitch-yaw' in launch_text
        or ('--roll' in launch_text
            and '--pitch' in launch_text
            and '--yaw' in launch_text)
    )

    readme_text = (ROOT / 'README.md').read_text()
    for phrase in [
        'wrap-around',
        'rigid',
        'JPEG',
        'checksum',
        'DiagnosticArray',
    ]:
        assert phrase.lower() in readme_text.lower()
