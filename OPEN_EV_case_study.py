#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Electric Vehicle Smart Charging case study considers the smart charging of
EVs within an unbalanced three-phase distribution network.

The case study considers a business park where 80 EVs are charged at 6.6 kW
charge points.

The objective is to charge all of the vehicles to their maximum energy level
prior to departure, at lowest cost.
"""

#import modules
import os
from os.path import normpath, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from System.Network_3ph_pf import Network_3ph
import System.Assets as AS
import System.Markets as MK
import System.EnergySystem as ES

############## VERSION ##############

__version__ = "1.0.0"
        
################################################
###       
### Case Study: Electric Vehicle Smart Charging
###        
################################################
 
################################################
### RUN OPT OR JUST PLOT (IF RESULTS PICKLED)
################################################

# run all optimisation/heuristic strategies
run_opt = 1
# list of strategies to evaluate
# Added heuristic strategies 'tou' (time-of-use rule) and 'valley'
# (valley-filling greedy)
# opt_type = ['open_loop', 'mpc', 'uncontrolled', 'edf', 'tou', 'valley', 'lp']
opt_type = ['open_loop', 'mpc', 'uncontrolled', 'edf', 'tou', 'valley', 'lp']


path_string = normpath('Results/EV_Case_Study/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)
save_suffix = '.pdf'

# folder for performance metric plots
metrics_path = join(path_string, 'performance_metrics')
if not os.path.isdir(metrics_path):
    os.makedirs(metrics_path)

# containers for performance metrics collected for each strategy
metrics = {
    'peak_import_power': {},
    'peak_energy_demand': {},
    'aggregate_waiting_time': {},
    'waiting_times': {},  # list of waiting times per EV
    'energy_deficits': {},  # energy deficit per EV at departure (kWh)
    'aggregate_energy_deficit': {},
    'energy_variability': {},
    'total_energy_cost': {},
    'cost_per_ev': {},
    'avg_cost_per_ev': {}
}

def figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                Pnet_market, storage_assets, N_ESs,\
                nondispatch_assets, time_ems, time, timeE, buses_Vpu):   
    
    # plot half hour predicted and actual net load
    title = '' #str(x)
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time_ems,P_demand_base_pred_ems,label=\
             'Predicted net load, 30 mins')
    plt.plot(time_ems,P_compare, label =\
             'Predicted net load + EVs charging, 30 mins')
    plt.ylabel('Power (kW)')
    plt.ylim(0, 2100)
    plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time_ems))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_ems_'  + str(x) + save_suffix)),
                bbox_inches='tight')

    # plot 5 minute predicted and actual net load
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time,P_demand_base,'--',label=\
             'Base Load')
    plt.plot(time,Pnet_market,label=\
             'Import Power')
    plt.ylabel('Power (kW)')
    plt.ylim(500, 2100)
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_actual_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot power for EV charging
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(time,sum(storage_assets[i].Pnet for i in range(N_ESs)))
    for i in range(N_EVs):
        plt.plot(time,storage_assets[i].Pnet)
    plt.xlim(0,24)
    plt.ylim(0,10)
    plt.ylabel('Power (kW)')
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_EVs_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot average battery energy
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(timeE,sum(storage_assets[i].E for i in range(N_ESs))/N_EVs)
    plt.ylabel('Average EV Energy (kWh)')
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.yticks(np.arange(0,37,4))
    plt.ylim(12, 36)
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('E_EVs_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot line voltages
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,0]>0,0],1),'-',label='Phase A')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,1]>0,1],1),'--',label='Phase B')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,2]>0,2],1),'-.',label='Phase C')
    plt.hlines(0.95,0,24,'r',':','Lower Limit')
    plt.ylabel('Minimum Voltage Mag. (pu)')
    plt.ylim(0.94, 1.00)
    plt.yticks(np.arange(0.95, 1.00, step=0.01))
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('Vmin_'  + str(x) + save_suffix)),
                bbox_inches='tight')

def record_metrics(strategy, storage_assets, P_import, P_demand,
                   ta_EVs, td_EVs, dt, dt_ems, Emax_EV, market):
    """Collect performance metrics for a strategy at system resolution.

    The metrics include peak import power, peak demand, waiting times,
    aggregate energy deficit, and an energy variability metric calculated as
    the difference between the maximum imported power and the minimum demand.

    Parameters
    ----------
    strategy : str
        Name of the control strategy being evaluated.
    storage_assets : list
        List of storage assets (EVs) in the system.
    P_import : ndarray
        Net power imported from the grid at the simulation time step (kW).
    P_demand : ndarray
        System power demand at the simulation time step (kW).
    ta_EVs, td_EVs : ndarray
        EV arrival and departure times (EMS resolution).
    dt, dt_ems : float
        Simulation and EMS time step sizes in hours.
    Emax_EV : float
        Maximum energy capacity of the EV batteries (kWh).
    market : MK.Market
        Market instance containing time-varying prices.
    """
    N_EVs = len(storage_assets)
    t_a = (ta_EVs * dt_ems / dt).astype(int)
    t_d = (td_EVs * dt_ems / dt).astype(int)

    waiting_times = []
    energy_deficits = []
    for i in range(N_EVs):
        power_i = storage_assets[i].Pnet
        arrival = t_a[i]
        departure = min(t_d[i], len(power_i))
        charging_idx = np.where(power_i[arrival:departure] > 0)[0]
        if charging_idx.size == 0:
            waiting_times.append((departure - arrival) * dt)
        else:
            waiting_times.append(charging_idx[0] * dt)

        energy_i = storage_assets[i].E
        departure_energy = energy_i[min(departure, len(energy_i) - 1)]
        energy_deficits.append(max(Emax_EV - departure_energy, 0))

    # Peak metrics computed at the finest simulation resolution
    metrics['peak_import_power'][strategy] = np.max(P_import)
    metrics['peak_energy_demand'][strategy] = np.max(P_demand)
    metrics['energy_variability'][strategy] = np.max(P_import) - np.min(P_demand)
    metrics['waiting_times'][strategy] = waiting_times
    metrics['aggregate_waiting_time'][strategy] = np.nansum(waiting_times)
    metrics['energy_deficits'][strategy] = energy_deficits
    metrics['aggregate_energy_deficit'][strategy] = np.nansum(energy_deficits)

    # Costs using market prices
    total_cost = -market.calculate_revenue(P_import, dt)
    costs_per_ev = [-market.calculate_revenue(sa.Pnet, dt) for sa in storage_assets]
    metrics['total_energy_cost'][strategy] = total_cost
    metrics['cost_per_ev'][strategy] = costs_per_ev
    metrics['avg_cost_per_ev'][strategy] = np.nanmean(costs_per_ev) if costs_per_ev else np.nan


def plot_performance_metrics(metrics, path):
    """Create comparative plots for collected metrics."""
    strategies = list(metrics['peak_import_power'].keys())
    if not strategies:
        return

    print("Performance metrics:")
    for s in strategies:
        print(f"Strategy '{s}':")
        peak_import = metrics['peak_import_power'][s]
        peak_demand = metrics['peak_energy_demand'][s]
        additional_import = peak_import - peak_demand
        energy_var = metrics['energy_variability'][s]
        print(f"  Peak Import Power: {peak_import} kW")
        print(f"  Peak Energy Demand: {peak_demand} kW")
        print(f"  Additional Imported Power: {additional_import} kW")
        print(f"  Energy Variability: {energy_var} kW")
        aggregate_wait = metrics['aggregate_waiting_time'][s]
        print(f"  Aggregate Waiting Time: {aggregate_wait} h")
        waiting_times = metrics['waiting_times'][s]
        if waiting_times:
            avg_wait = np.nanmean(waiting_times)
            print(f"  Average Waiting Time per EV: {avg_wait} h")
        energy_deficits = metrics['energy_deficits'][s]
        if energy_deficits:
            avg_deficit = np.nanmean(energy_deficits)
            agg_deficit = metrics['aggregate_energy_deficit'][s]
            print(f"  Average Energy Deficit at Departure: {avg_deficit} kWh")
            print(f"  Aggregate Energy Deficit: {agg_deficit} kWh")
        total_cost = metrics['total_energy_cost'][s]
        print(f"  Total Cost: AUD {total_cost}")
        avg_ev_cost = metrics['avg_cost_per_ev'][s]
        if not np.isnan(avg_ev_cost):
            print(f"  Average Cost per EV: AUD {avg_ev_cost}")
        print()

    def bar_plot(values_dict, ylabel, filename):
        vals = [values_dict[s] for s in strategies]
        plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
        plt.bar(strategies, vals)
        plt.ylabel(ylabel)
        plt.xlabel('Strategy')
        plt.tight_layout()
        plt.savefig(join(path, normpath(filename + save_suffix)), bbox_inches='tight')
        plt.close()

    bar_plot(metrics['peak_import_power'], 'Peak Import Power (kW)',
             'peak_import_power')
    bar_plot(metrics['peak_energy_demand'], 'Peak Energy Demand (kW)',
             'peak_energy_demand')
    # Combined stacked bar: base demand and additional imported power
    import_minus_demand = {
        s: metrics['peak_import_power'][s] - metrics['peak_energy_demand'][s]
        for s in strategies
    }
    demand_vals = [metrics['peak_energy_demand'][s] for s in strategies]
    extra_vals = [import_minus_demand[s] for s in strategies]
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w',
               edgecolor='k')
    plt.bar(strategies, demand_vals, label='Peak Energy Demand')
    plt.bar(strategies, extra_vals, bottom=demand_vals,
            label='Additional Imported Power', color='orange')
    plt.ylabel('Power (kW)')
    plt.xlabel('Strategy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(path, normpath('import_power_combined' + save_suffix)),
                bbox_inches='tight')
    plt.close()

    # Only the additional imported power (effect of EVs)
    bar_plot(import_minus_demand, 'Additional Imported Power (kW)',
             'import_power_minus_demand')
    bar_plot(metrics['energy_variability'], 'Energy Variability (kW)',
             'energy_variability')
    bar_plot(metrics['aggregate_waiting_time'], 'Aggregate Waiting Time (h)',
             'aggregate_waiting_time')
    bar_plot(metrics['aggregate_energy_deficit'], 'Aggregate Energy Deficit (kWh)',
             'aggregate_energy_deficit')
    bar_plot(metrics['avg_cost_per_ev'], 'Average Cost per EV (AUD)',
             'avg_cost_per_ev')
    bar_plot(metrics['total_energy_cost'], 'Total Cost (AUD)',
             'total_cost')

    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.boxplot([metrics['waiting_times'][s] for s in strategies], labels=strategies)
    plt.ylabel('Waiting Time per EV (h)')
    plt.tight_layout()
    plt.savefig(join(path, normpath('waiting_time_per_ev' + save_suffix)),
                bbox_inches='tight')
    plt.close()

    data = []
    labels = []
    for s in strategies:
        arr = np.array(metrics['energy_deficits'][s], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            data.append(arr)
            labels.append(s)
    if data:
        plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w',
                   edgecolor='k')
        plt.boxplot(data, labels=labels)
        plt.ylabel('Energy Deficit at Departure (kWh)')
        plt.tight_layout()
        plt.savefig(join(path, normpath('energy_deficit_per_ev' + save_suffix)),
                    bbox_inches='tight')
        plt.close()


if run_opt ==1:
           
    #######################################
    ### STEP 0: Load Data
    #######################################
            
    PV_data_path = os.path.join("Data", "PVpu_1min.csv")    
    PVpu_raw = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values[:,0]
    
    substation_data = pickle.load(open(os.path.join\
                                       ("Data","substation_daily_PQ_data.p"),'rb'))
    T_5min_sub_data = substation_data[0]
    N_days_sub_data = substation_data[1]
    P_raw_days_sub_data = 1*substation_data[2]
    Q_raw_days_sub_data = 1*substation_data[3]
    
    #######################################
    ### STEP 1: setup parameters
    #######################################
    
    dt = 5/60  # 5 minute time intervals
    T = int(24/dt)  # Number of time intervals
    dt_ems = 30/60  # 30 minute EMS time intervals
    T_ems = int(T*dt/dt_ems)  # Number of EMS intervals
    T0 = 0  # from 12 am to 12 am
    N_PVs = 5  # Number of PVs
    P_pv = 200*np.ones(N_PVs)  # PV rated power (kW)
    PV_bus_names = ['634','645','652','671','675']
    PV_phases = [[0,1,2],[1],[0],[0,1,2],[0,1,2]]  # phases at each bus
    
    # Electric Vehicle (EV) parameters
    N_EVs = 80  # number of EVs
    #N_EVs = 2  # number of EVs
    Emax_EV = 36  # maximum EV energy level
    Emin_EV = 0  # minimum EV energy level
    P_max_EV = 6.6  # maximum EV charging power
    P_min_EV = 0  # minimum EV charging power
    
    # EV charge/discharge efficiency
    eff_EV = np.ones(100)
    eff_EV[0:50] = 0.6
    eff_EV[50:70] = 0.75
    eff_EV[70:100] = 0.8
    eff_EV_opt = 1  # fixed efficiency for EVs to use in optimiser
    
    # EV arrival & departure times and energy levels on arrival
    np.random.seed(1000)
    # random EV initial energy levels
    E0_EVs = Emax_EV*np.random.uniform(0.2,0.9,N_EVs)
    # random EV arrival times between 6am and 9am
    ta_EVs = np.random.randint(int(6/dt_ems),int(10/dt_ems),N_EVs) - int(T0/dt_ems)
    # random EV departure times between 5pm and 9pm
    td_EVs = np.random.randint(int(15/dt_ems),\
                               int(21/dt_ems),N_EVs) - int(T0/dt_ems)
    
    # Market parameters
    # market and EMS have the same time-series
    dt_market = dt_ems
    T_market = T_ems
    
    # Import and Export Prices
    # price_df = pd.read_csv(
    #     "NEMPRICEANDDEMAND_NSW1_202508231510.csv",
    #     parse_dates=["Settlement Date"],
    #     dayfirst=True,
    # ).set_index("Settlement Date")

    # start = price_df.index[0].normalize() + pd.Timedelta(days=1)
    # end = start + pd.Timedelta(hours=23, minutes=55)
    # day_prices = price_df.loc[start:end]

    # spot = day_prices["Spot Price ($/MWh)"].to_numpy() / 1000  # AUD/kWh
    # n_per_market = int(dt_market / dt)
    # required = T_market * n_per_market
    # if len(spot) < required:
    #     raise ValueError("Insufficient spot price data for simulation horizon")
    # prices_import = spot.reshape(T_market, n_per_market).mean(axis=1)
    # prices_export = prices_import.copy()
    # demand_charge = 0.1  # (AUD/kW) for the maximum demand

    # Original fixed-price strategy (GBP) for reference
    prices_export = 0.095*np.ones(T_market)  #(£/kWh)
    prices_import = 0.285*np.ones(T_market)  #(£/kWh)
    demand_charge = 0.19  # (£/kW) for the maximum demand
    
    # Site Power Constraints
    Pmax_market = 100e3*np.ones(T_market)
    Pmin_market = -100e3*np.ones(T_market)
    
    # PV data set up
    N_sub_data = P_raw_days_sub_data.shape[1]
    P_sub0 = np.zeros([T,N_sub_data])
    Q_sub0 = np.zeros([T,N_sub_data])
    P_sub = np.zeros([T,N_sub_data])
    Q_sub = np.zeros([T,N_sub_data])
    dt_raw = 1/60  # 1 minute time intervals
    T_raw = int(24/dt_raw)  # Number of data time intervals
    dt_sub_raw = 5/60  # 5 minute time intervals
    T_sub_raw = int(24/dt_raw)  # Number of data time intervals
    PVpu_8am = np.zeros(T)
    for t in range(T):
        t_raw_indexes = (t*dt/dt_raw + np.arange(0,dt/dt_raw)).astype(int)
        t_sub_indexes = (t*dt/dt_sub_raw + np.arange(0,dt/dt_sub_raw)).astype(int)
        PVpu_8am[t] = np.mean(PVpu_raw[t_raw_indexes])
        P_sub[t,:] = np.mean(P_raw_days_sub_data[t_sub_indexes,:],0)
        Q_sub[t,:] = np.mean(Q_raw_days_sub_data[t_sub_indexes,:],0)
    
    # Shift PV to 12am from 8am start time
    PVpu = np.zeros(T)
    for t in range(T):
        t_sub0 = int((t-8/dt)%T)
        PVpu[t] = PVpu_8am[t_sub0]
    
    #######################################
    ### STEP 2: setup the network
    #######################################
    
    # from https://github.com/e2nIEE/pandapower/blob/
    # master/tutorials/minimal_example.ipynb
    network = Network_3ph()  # IEEE 13 bus by default
    network.capacitor_df = network.capacitor_df[0:0] #removes the capacitors
    network.update_YandZ()
    
    # set bus voltage limits
    network.set_pf_limits(0.95*network.Vslack_ph, 1.05*network.Vslack_ph,
                          2000e3/network.Vslack_ph)
    
    # set up busses
    bus650_num = network.bus_df[network.bus_df['name']=='650'].number.values[0]
    bus634_num = network.bus_df[network.bus_df['name']=='634'].number.values[0]
    bus645_num = network.bus_df[network.bus_df['name']=='645'].number.values[0]
    bus646_num = network.bus_df[network.bus_df['name']=='646'].number.values[0]
    bus652_num = network.bus_df[network.bus_df['name']=='652'].number.values[0]
    bus671_num = network.bus_df[network.bus_df['name']=='671'].number.values[0]
    bus675_num = network.bus_df[network.bus_df['name']=='675'].number.values[0]
    phase_array = np.array([0,1,2])
    N_buses = network.N_buses  # Number of buses
    N_phases = network.N_phases  # Number of phases
    N_load_bus_phases = N_phases*(N_buses-1)  # Number of load buses 
    N_lines = network.N_lines  # Number lines
    N_line_phases = N_lines*N_phases 
    
    
    #######################################
    ### STEP 3: setup the assets 
    #######################################
    
    storage_assets = []
    nondispatch_assets = []
    smooth = True
    
    # Method to smooth actual data to generate equivalent of predicted data
    def smoothing(Pnet, Qnet):
        h = 20
        m = len(Pnet)
        Pnet_pred = np.zeros(m)
        Qnet_pred = np.zeros(m)
        P_cont = np.tile(Pnet,2)
        Q_cont = np.tile(Qnet,2)
        for i in range(m):
            Pnet_pred[i] = sum(P_cont[i:i+h])/(h) 
            Qnet_pred[i] = sum(Q_cont[i:i+h])/(h)
        return{"Pnet_pred": Pnet_pred, "Qnet_pred": Qnet_pred}
    
    # Create loads
    sub_load_index = 0
    # Create loads at bus 634
    for ph_i in range(3):
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = None
        Qnet_pred = None
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus634_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 645
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus645_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 646
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus645_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 652
    for ph_i in [0]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus652_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 671
    for ph_i in range(3):
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus671_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    # Create loads at bus 675 (3->a, 1->b, 2->c)
    for ph_i in [0]:
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    for ph_i in [2]:
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    
    # Add PV generation sources
    for i in range(N_PVs):
        Pnet_i = -PVpu*P_pv[i]
        Qnet_i = np.zeros(T)
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet_i,Qnet_i)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        bus_id_i = network.bus_df[network.bus_df['name']==\
                                  PV_bus_names[i]].number.values[0]
        phases_i = PV_phases[i]
        PV_gen_i = AS.NondispatchableAsset_3ph(Pnet_i, Qnet_i, bus_id_i, phases_i,
                                               dt, T, Pnet_pred = Pnet_pred,
                                               Qnet_pred = Qnet_pred)
        nondispatch_assets.append(PV_gen_i)
    N_NDE = len(nondispatch_assets)
    
    # EVs at bus 634
    for i in range(N_EVs): 
        Emax_ev_i = Emax_EV*np.ones(T_ems)
        Emin_ev_i = Emin_EV*np.ones(T_ems)
        Pmax_ev_i = np.zeros(T_ems)
        Pmin_ev_i = np.zeros(T_ems)
        for t in range(ta_EVs[i],int(min(td_EVs[i],T_ems))):
            Pmax_ev_i[t] = P_max_EV
            Pmin_ev_i[t] = P_min_EV
        bus_id_ev_i = bus634_num
        ev_i = AS.StorageAsset(Emax_ev_i, Emin_ev_i, Pmax_ev_i, Pmin_ev_i,
                               E0_EVs[i], Emax_EV, bus_id_ev_i, dt, T, dt_ems,
                               T_ems, Pmax_abs=P_max_EV, c_deg_lin = 0,
                               eff = eff_EV, eff_opt = eff_EV_opt)
        storage_assets.append(ev_i)
    N_ESs = len(storage_assets)
        
    #######################################
    ### STEP 4: setup the market
    #######################################
        
    bus_id_market = bus650_num
    market = MK.Market(bus_id_market, prices_export, prices_import, demand_charge,
                       Pmax_market, Pmin_market, dt_market, T_market)
    
    #######################################
    #STEP 5: setup the energy system
    #######################################
    
    energy_system = ES.EnergySystem(storage_assets, nondispatch_assets, network,
                                    market, dt, T, dt_ems, T_ems)

    # pre-compute base demand profiles for use in heuristic controllers
    P_demand_base = np.zeros(T)  # actual base demand at system resolution
    P_demand_base_pred = np.zeros(T)  # predicted base demand
    for nd in nondispatch_assets:
        P_demand_base += nd.Pnet
        P_demand_base_pred += nd.Pnet_pred

    #######################################
    ### STEP 6: simulate the energy system:
    #######################################
    
    i_line_unconst_list = list(range(network.N_lines))
    v_bus_unconst_list = []
    
    for x in opt_type:
        if x == "open_loop":
            output = energy_system.\
                    simulate_network_3phPF('3ph',\
                                           i_unconstrained_lines=\
                                           i_line_unconst_list,\
                                           v_unconstrained_buses=\
                                           v_bus_unconst_list)

        if x == "mpc":
            output = energy_system.\
                    simulate_network_mpc_3phPF('3ph',
                                               i_unconstrained_lines=\
                                               i_line_unconst_list,\
                                               v_unconstrained_buses=\
                                               v_bus_unconst_list)

        if x == "uncontrolled": 
            P_ESs = np.zeros((T, N_ESs)) #create array to EV charging power at each timestep
            for i in range(N_EVs): # for every EV
                t_a = int(ta_EVs[i] * dt_ems / dt) #arrival time
                t_d = int(td_EVs[i] * dt_ems / dt) # departure time
                E = E0_EVs[i] # EV stored energy at its initial state
                t = t_a
                while t < t_d and E < Emax_EV: # while EV is here and not full, keep charging
                    P_ESs[t, i] = P_max_EV # EV charge is at max
                    E += P_max_EV * dt
                    t += 1
            output = energy_system.simulate_network_manual_dispatch(P_ESs)

        if x == "edf":
            P_ESs = np.zeros((T, N_ESs))
            E_state = E0_EVs.copy() #copy of each EV and its state of charge
            t_a_dt = (ta_EVs * dt_ems / dt).astype(int) #time arrival/departure
            t_d_dt = (td_EVs * dt_ems / dt).astype(int)
            for t in range(T):
                t_ems = int(t / (dt_ems / dt))
                P_avail = max(market.Pmax[t_ems] - P_demand_base[t], 0) #power available in the market
                connected = [i for i in range(N_EVs)
                             if t_a_dt[i] <= t < t_d_dt[i] and E_state[i] < Emax_EV] #list of all connected EVs at current moment needing charge
                connected.sort(key=lambda i: t_d_dt[i]) # sort by earliest departure
                for i in connected: # loop through sorted list of connected EVs
                    P_need = (Emax_EV - E_state[i]) / dt # power needed to finish charging in one step if possib;e
                    P_ch = min(P_max_EV, P_avail, P_need) # must limit charging power
                    if P_ch <= 0: # skip if no power available to charge
                        continue
                    P_ESs[t, i] = P_ch
                    E_state[i] += P_ch * dt
                    P_avail -= P_ch
                if P_avail <= 0:
                    break
            output = energy_system.simulate_network_manual_dispatch(P_ESs)

        if x == "tou":
            # Time-of-Use heuristic: charge at max power during off-peak
            # periods or when close to departure.
            P_ESs = np.zeros((T, N_ESs)) # matrix to store every EV at each time step
            E_state = E0_EVs.copy()
            t_a_dt = (ta_EVs * dt_ems / dt).astype(int) # arrival and departure time conversion to be compaatible with sim
            t_d_dt = (td_EVs * dt_ems / dt).astype(int)
            safeguard = 2  # hours before departure to start charging, to avoid non-charges
            for t in range(T):
                hour = t * dt # current hour
                off_peak = (hour < 7) or (hour >= 22) # offpeak range
                t_ems = int(t / (dt_ems / dt))
                P_avail = max(
                    market.Pmax[t_ems] - P_demand_base[t] - P_ESs[t].sum(), 0
                )
                for i in range(N_EVs): #iterate over all evs
                    if t_a_dt[i] <= t < t_d_dt[i] and E_state[i] < Emax_EV: # only consider evs that are present and not fully charged
                        hrs_to_depart = (t_d_dt[i] - t) * dt
                        if off_peak or hrs_to_depart <= safeguard: # if true then charge
                            P_need = (Emax_EV - E_state[i]) / dt
                            P_ch = min(P_max_EV, P_avail, P_need) 
                            if P_ch <= 0:
                                continue
                            P_ESs[t, i] = P_ch
                            E_state[i] += P_ch * dt
                            P_avail -= P_ch
                            if P_avail <= 0:
                                break
            output = energy_system.simulate_network_manual_dispatch(P_ESs)

        if x == "valley":
            # Valley-filling greedy heuristic
            P_ESs = np.zeros((T, N_ESs))
            t_a_dt = (ta_EVs * dt_ems / dt).astype(int)
            t_d_dt = (td_EVs * dt_ems / dt).astype(int)
            current_load = P_demand_base_pred.copy()
            for i in range(N_EVs):
                energy_needed = Emax_EV - E0_EVs[i] #how much energy required to be full
                available = np.arange(t_a_dt[i], t_d_dt[i]) # when the EV is present
                # sort available times by current total load (low to high)
                sorted_times = sorted(available, key=lambda tt: current_load[tt]) # rank the timeslots to fill valleys first
                for t in sorted_times:
                    if energy_needed <= 0:
                        break
                    t_ems = int(t / (dt_ems / dt))
                    P_avail = max(market.Pmax[t_ems] - current_load[t], 0)
                    if P_avail <= 0:
                        continue
                    P_ch = min(P_max_EV, P_avail, energy_needed / dt)
                    P_ESs[t, i] = P_ch
                    current_load[t] += P_ch
                    energy_needed -= P_ch * dt
            output = energy_system.simulate_network_manual_dispatch(P_ESs)

        if x == "lp":
            for nd in nondispatch_assets: # for every non-dispatachable asset in the network, set predicited power to actual power for whole day
                nd.Pnet_pred = nd.Pnet.copy()
            P_demand_base_pred = P_demand_base.copy() # perfect forecast
            # run 3-phase power flow with no network congestion
            output = energy_system.simulate_network_3phPF(
                "copper_plate",
                i_unconstrained_lines=i_line_unconst_list,
                v_unconstrained_buses=v_bus_unconst_list,
            )

        PF_network_res = output['PF_network_res']
        P_ES_ems = output['P_ES_ems']
        
        Pnet_market = np.zeros(T)
        for t in range(T):
            market_bus_res = PF_network_res[t].res_bus_df.iloc[bus_id_market]
            # sum the three-phase complex powers, take real net power at time t
            Pnet_market[t] = np.real(
                market_bus_res["Sa"]
                + market_bus_res["Sb"]
                + market_bus_res["Sc"]
            )

        # system demand at simulation resolution (base + EV charging)
        P_demand = P_demand_base.copy()
        for sa in storage_assets:
            P_demand += sa.Pnet

        record_metrics(x, storage_assets, Pnet_market, P_demand,
                        ta_EVs, td_EVs, dt, dt_ems, Emax_EV, market)
        
        buses_Vpu = np.zeros([T,N_buses,N_phases])
        for t in range(T):
            for bus_id in range(N_buses):
                bus_res = PF_network_res[t].res_bus_df.iloc[bus_id]
                buses_Vpu[t,bus_id,0] = np.abs(bus_res['Va'])/network.Vslack_ph        
                buses_Vpu[t,bus_id,1] = np.abs(bus_res['Vb'])/network.Vslack_ph                  
                buses_Vpu[t,bus_id,2] = np.abs(bus_res['Vc'])/network.Vslack_ph         
        
        P_demand_base_pred_ems = np.zeros(T_ems)
        for t_ems in range(T_ems):
            t_indexes = (t_ems*dt_ems/dt + np.arange(0,dt_ems/dt)).astype(int)
            P_demand_base_pred_ems[t_ems] = np.mean(P_demand_base_pred[t_indexes])
        
        EVs_tot = sum(P_ES_ems[:,n] for n in range(N_ESs))
        P_compare = P_demand_base_pred_ems + EVs_tot
        #######################################
        ### STEP 7: plot results
        #######################################
        
        #x-axis time values
        time = dt*np.arange(T)
        time_ems = dt_ems*np.arange(T_ems)
        timeE = dt*np.arange(T+1)
        
        #energy cost
        energy_cost = market.calculate_revenue(Pnet_market,dt)
        print(f'Total energy cost: AUD {-1*energy_cost}')
        
        #save the data
        if x == "open_loop":
            pickled_data_OL = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_OL, open(join(path_string, normpath("EV_case_data_open_loop.p")), "wb"))

        if x == "mpc":
            pickled_data_MPC = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_MPC, open(join(path_string, normpath("EV_case_data_mpc.p")), "wb"))

        if x == "uncontrolled":
            pickled_data_UC = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_UC, open(join(path_string, normpath("EV_case_data_uncontrolled.p")), "wb"))

        if x == "edf":
            pickled_data_EDF = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_EDF, open(join(path_string, normpath("EV_case_data_edf.p")), "wb"))

        if x == "tou":
            pickled_data_TOU = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_TOU, open(join(path_string, normpath("EV_case_data_tou.p")), "wb"))

        if x == "valley":
            pickled_data_VF = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_VF, open(join(path_string, normpath("EV_case_data_valley.p")), "wb"))

        if x == "lp":
            pickled_data_LP = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_LP, open(join(path_string, normpath("EV_case_data_lp.p")), "wb"))
        
        
        figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                    Pnet_market, storage_assets, N_ESs,\
                    nondispatch_assets, time_ems, time, timeE, buses_Vpu)
    plot_performance_metrics(metrics, metrics_path)

# Load pickled data and plot
else:
    for x in opt_type:
        if x == "open_loop":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_open_loop.p")), "rb"))

        if x == "mpc":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_mpc.p")), "rb"))

        if x == "uncontrolled":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_uncontrolled.p")), "rb"))

        if x == "edf":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_edf.p")), "rb"))

        if x == "tou":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_tou.p")), "rb"))

        if x == "valley":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_valley.p")), "rb"))

        if x == "lp":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_lp.p")), "rb"))

        N_EVs = import_data[0]
        P_demand_base_pred_ems = import_data[1]
        P_compare = import_data[2]
        P_demand_base = import_data[3]
        Pnet_market = import_data[4]
        storage_assets = import_data[5]
        N_ESs = import_data[6]
        nondispatch_assets = import_data[7]
        time_ems = import_data[8]
        time = import_data[9]
        timeE = import_data[10]
        buses_Vpu = import_data[11]
        
        figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                Pnet_market, storage_assets, N_ESs,\
                nondispatch_assets, time_ems, time, timeE, buses_Vpu)



    
