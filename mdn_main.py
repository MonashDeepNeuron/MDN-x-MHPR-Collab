import argparse
import numpy as np
import gc
from tqdm import tqdm

import Modules.MonteCarloTools as MCTools
from Modules.run_monte_carlo_sweep import MonteCarlo
from Modules.Structures import CoordinateSystemType, FrameType
from Modules.Sensors import TVCAction
from Modules.main import Simulation

def has_reached_burnout(sim: Simulation):
    return (np.abs(sim.propulsion.getForce(sim.state)[0][0]) == 0) and (sim.state.time > sim.state.dt)

def run_mdn_sim(mc: MonteCarlo, inputs: tuple) -> None:
    name, config, i = inputs
    sim = mc._initialise_simulation(name, config, i)

    # Initial states
    sBI__G = sim.state.getPosition(CoordinateSystemType.GEOCENTRIC)
    T_gG = sim.state.getTransformationMatrix(CoordinateSystemType.GEOGRAPHIC, CoordinateSystemType.GEOCENTRIC)
    sBI__G_initial = sim.state.getPosition(CoordinateSystemType.GEOCENTRIC)
    initial_alt = sim.state.getAltitudeGeometric()

    step_count = 0
    # We terminate the sim at burnout since TVC loses control authority beyond this point (and hence RL receives poor gradients)
    while sim.should_continue_loop(sim.state) and not has_reached_burnout(sim):
        # Only giving the model access every few timesteps (based on the hardware's clockspeed) and prior to burnout
        if step_count % int(1/(sim.tvc_clockspeed*sim.state.dt)) == 0:
            # Returns current state info
            #TODO IN THE EARTH FRAME - IRL MAY BE IN THE ROCKET'S FRAME (e.g. acceleration measured on board)
            altitude = sim.state.getAltitudeGeometric() - initial_alt # Zeroed for RL
            sBI__G = sim.state.getPosition(CoordinateSystemType.GEOCENTRIC)
            displacement = T_gG @ (sBI__G - sBI__G_initial)
            vB_E_B = sim.state.getVelocity(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)
            aB_E_B = sim.state.getAcceleration(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)
            euler_angles = np.rad2deg(sim.state.euler_angles)
            wB_E_B = sim.state.getAngularVelocity(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)

            # Agent chooses action
            action = None #TVCAction.TILT_UP # @MDN THIS IS WHERE YOU PASS IN YOUR ACTION
            if action == TVCAction.TILT_UP:
                sim.tilt += sim.tilt_stepsize
            elif action == TVCAction.TILT_DOWN:
                sim.tilt -= sim.tilt_stepsize
            elif action == TVCAction.PAN_UP:
                sim.pan += sim.pan_stepsize
            elif action == TVCAction.PAN_DOWN:
                sim.pan -= sim.pan_stepsize
            else:
                pass
                #raise ValueError(f'Action {action} not recognised. Please use TU, TD, PU, PD')

            # Enforcing a hard limit on tilt and pan
            if sim.tilt > sim.max_tilt:
                sim.tilt -= sim.tilt_stepsize
            elif sim.tilt < -sim.max_tilt:
                sim.tilt += sim.tilt_stepsize
            if sim.pan > sim.max_pan:
                sim.pan -= sim.pan_stepsize
            elif sim.pan < -sim.max_pan:
                sim.pan += sim.pan_stepsize

        # Updating the state given the action
        sim.iteration()
        step_count += 1

    # Keeping memory usage in check
    del sim.aerodynamics.force_coefficients
    del sim.aerodynamics.moment_coefficients
    try:
        del sim.atmosphere.pyatm
    except AttributeError:
        pass

    gc.collect()  # Runs the garbage collection.

    state_history, _ = sim.finalise_run()
    mc.results[sim] = state_history.df

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle_config', type=str, help='.yaml file which specifies the initial parameters', default='mdn_vehicleConfig.yaml')
    parser.add_argument('--mc_config', type=str, help='.yaml file which specifies the monte carlo parameters', default='mdn_mcConfig.yaml')
    parser.add_argument('--save_loc', type=str, help='Path to Monte Carlo save location. Not saved by default', default=None)
    args = parser.parse_args()

    mc = MonteCarlo(args.mc_config, args.vehicle_config)
    cases = mc._generate_cases()

    for inputs in tqdm(cases):
        run_mdn_sim(mc, inputs)

    # Saving sim data for postprocessing
    if args.save_loc is not None:
        res = MCTools.MonteCarloResults(mc.results)
        res.save(f'{args.save_loc}.pkl')

    import utils
    print(utils.get_open_file_count())
