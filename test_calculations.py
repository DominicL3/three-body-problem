import numpy as np
import three_body

def test_simple_forces():
    body1 = three_body.Body(1/6.67e-11, np.array([0, 0, 0]), np.array([0, 0, 0]))
    body2 = three_body.Body(100, np.array([0, 0, 10]), np.array([0, 0, 0]))
    F12 = body1.calculate_force([body2]) # force on body1 from body2
    F21 = body2.calculate_force([body1]) # force on body2 from body1

    # check if vector is correct
    assert (F12 == np.array([0, 0, 1])).all()
    assert (F21 == np.array([0, 0, -1])).all()

def test_simple_position_update():
    body1 = three_body.Body(1/6.67e-11, np.array([0, 0, 0]), np.array([0, 0, 0]))
    body2 = three_body.Body(100, np.array([0, 0, 10]), np.array([0, 0, 0]))

    body2.update_position([body1], 100)
    assert np.allclose(body2.position, np.array([0, 0, -90]))

def test_simple_timestep_update():
    body1 = three_body.Body(1/6.67e-11, np.array([0, 0, 0]), np.array([0, 0, 0]))
    body2 = three_body.Body(100, np.array([0, 0, 10]), np.array([0, 0, 0]))

    three_body.update_timestep([body1, body2], 100)
    assert np.allclose(body2.position, np.array([0, 0, -90]))

def test_circular_motion_2body():
    sun = three_body.Body(1/6.67e-11, np.array([0, 0, 0]), np.array([0, 0, 0]))
    pebble = three_body.Body(1, np.array([100, 0, 0]), np.array([0, 1/10, 0]))

    # let the clock run a bit
    for i in np.arange(10000):
        three_body.update_timestep([sun, pebble], 1)

    # ensure radius and velocity is still about the same in circular case
    radius = np.linalg.norm(sun.position - pebble.position)
    mag_velocity = np.linalg.norm(pebble.velocity)

    print(f"New pebble position: {pebble.position}")

    assert np.allclose(radius, 100, 0.1), f"Radius has changed from 100 to {radius}"
    assert np.allclose(mag_velocity, 1/10, 0.1), f"Velocity has changed from 1/10 to {mag_velocity}"