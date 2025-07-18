from .appliances_programs import TumbleDryer, WashingMachine
set_appliances = \
{
  "FridgeFreezer": {
    "frad": 0.46, 
    "cycle_power": 190, 
    "cal": 0.05348400435732094, 
    "standby_power": 0, 
    "owner": 0.651, 
    "cycle_n": 6115.75932851014, 
    "fconv": 0.64, 
    "name": "FridgeFreezer", 
    "cycle_length": 22, 
    "consumption": 426.064566552874, 
    "delay": 64, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "Hob": {
    "frad": 0.3, 
    "cycle_power": 2400, 
    "cal": 0.036523596330418226, 
    "standby_power": 1, 
    "owner": 0.463, 
    "cycle_n": 417.902007298679, 
    "fconv": 0.7, 
    "name": "Hob", 
    "cycle_length": 16, 
    "consumption": 276.105844135875, 
    "delay": 0, 
    "activity": "food", 
    "type": "appliance"
  }, 
  "Clock": {
    "frad": 0.5, 
    "cycle_power": 2, 
    "cal": 1.90258751902588e-11, 
    "standby_power": 2, 
    "owner": 0.9, 
    "cycle_n": 1e-05, 
    "fconv": 0.5, 
    "name": "Clock", 
    "cycle_length": 0, 
    "consumption": 17.52, 
    "delay": 0, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "DVD": {
    "frad": 0.92, 
    "cycle_power": 33.5521973529143, 
    "cal": 0.055821331844623616, 
    "standby_power": 2, 
    "owner": 0.896, 
    "cycle_n": 1464.26653737683, 
    "fconv": 0.08, 
    "name": "DVD", 
    "cycle_length": 73, 
    "consumption": 73.7310058969084, 
    "delay": 0, 
    "activity": "tv", 
    "type": "appliance"
  }, 
  "Kettle": {
    "frad": 0.17, 
    "cycle_power": 2000, 
    "cal": 0.004612679788676307, 
    "standby_power": 1, 
    "owner": 0.975, 
    "cycle_n": 1519.82286616642, 
    "fconv": 0.83, 
    "name": "Kettle", 
    "cycle_length": 3, 
    "consumption": 160.666295473334, 
    "delay": 0, 
    "activity": "Presence", 
    "type": "appliance"
  }, 
  "Microwave": {
    "frad": 0, 
    "cycle_power": 1250, 
    "cal": 0.008013148873682017, 
    "standby_power": 2, 
    "owner": 0.859, 
    "cycle_n": 94.619322407248, 
    "fconv": 0, 
    "name": "Microhave", 
    "cycle_length": 30, 
    "consumption": 76.5624571821228, 
    "delay": 0, 
    "activity": "food", 
    "type": "appliance"
  }, 
  "HiFi": {
    "frad": 0.5, 
    "cycle_power": 100, 
    "cal": 0.008005019574356842, 
    "standby_power": 9, 
    "owner": 0.9, 
    "cycle_n": 109.425544999843, 
    "fconv": 0.5, 
    "name": "HiFi", 
    "cycle_length": 60, 
    "consumption": 88.7977245949857, 
    "delay": 0, 
    "activity": "audio", 
    "type": "appliance"
  }, 
  "bathFlow": {
    "cycle_length": 10, 
    "cal": 0.01447147553646071, 
    "owner": 0.0, 
    "standby_flow": 0, 
    "activity": "shower", 
    "type": "tapping", 
    "cycle_n": 52, 
    "cycle_flow": 14
  }, 
  "ChestFreezer": {
    "frad": 0.46, 
    "cycle_power": 190, 
    "cal": 0.06627085372080074, 
    "standby_power": 0, 
    "owner": 0.163, 
    "cycle_n": 6115.75932851014, 
    "fconv": 0.64, 
    "name": "ChestFreezer", 
    "cycle_length": 14, 
    "consumption": 271.131996897283, 
    "delay": 72, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "Fax": {
    "frad": 0.5, 
    "cycle_power": 37, 
    "cal": 0.0005869916079454653, 
    "standby_power": 3, 
    "owner": 0.2, 
    "cycle_n": 197.123588348433, 
    "fconv": 0.5, 
    "name": "Fax", 
    "cycle_length": 31, 
    "consumption": 29.7428043686541, 
    "delay": 0, 
    "activity": "Presence", 
    "type": "appliance"
  }, 
  "Printer": {
    "frad": 0.67, 
    "cycle_power": 335.2, 
    "cal": 0.06347513729112045, 
    "standby_power": 4, 
    "owner": 0.665, 
    "cycle_n": 654.875958011369, 
    "fconv": 0.33, 
    "name": "Printer", 
    "cycle_length": 4, 
    "consumption": 49.499661152891, 
    "delay": 0, 
    "activity": "pc", 
    "type": "appliance"
  }, 
  "PC": {
    "frad": 0.92, 
    "cycle_power": 140.7, 
    "cal": 0.08083397129113995, 
    "standby_power": 5, 
    "owner": 0.708, 
    "cycle_n": 449.237984776012, 
    "fconv": 0.08, 
    "name": "PC", 
    "cycle_length": 300, 
    "consumption": 348.607972670524, 
    "delay": 0, 
    "activity": "pc", 
    "type": "appliance"
  }, 
  "CordlessPhone": {
    "frad": 0.5, 
    "cycle_power": 1, 
    "cal": 4.14823389814581e-11, 
    "standby_power": 1, 
    "owner": 0.9, 
    "cycle_n": 1e-05, 
    "fconv": 0.5, 
    "name": "CordlessPhone", 
    "cycle_length": 0, 
    "consumption": 8.76, 
    "delay": 0, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "WasherDryer": {
    "frad": 0.0, 
    "cycle_power": 792.034786057663, 
    "cal": 0.28709666854255933, 
    "standby_power": 1, 
    "owner": 0.153, 
    "cycle_n": 195.906595865545, 
    "fconv": 0.25, 
    "name": "WasherDryer", 
    "cycle_length": 198, 
    "consumption": 520.157476087695, 
    "delay": 0, 
    "activity": "washing", 
    "type": "appliance"
  }, 
  "UprightFreezer": {
    "frad": 0.46, 
    "cycle_power": 155, 
    "cal": 0.04577208372652669, 
    "standby_power": 0, 
    "owner": 0.291, 
    "cycle_n": 6115.75932851014, 
    "fconv": 0.64, 
    "name": "UprightFreezer", 
    "cycle_length": 20, 
    "consumption": 315.980898639691, 
    "delay": 66, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "Vacuum": {
    "frad": 0.5, 
    "cycle_power": 2000, 
    "cal": 0.14321421339450868, 
    "standby_power": 0, 
    "owner": 0.937, 
    "cycle_n": 110.367597918536, 
    "fconv": 0.5, 
    "name": "Vacuum", 
    "cycle_length": 20, 
    "consumption": 73.5783986123575, 
    "delay": 0, 
    "activity": "vacuum", 
    "type": "appliance"
  }, 
  "Refrigerator": {
    "frad": 0.46, 
    "cycle_power": 110, 
    "cal": 0.035861772069934156, 
    "standby_power": 0, 
    "owner": 0.43, 
    "cycle_n": 6115.75932851014, 
    "fconv": 0.64, 
    "name": "Refrigerator", 
    "cycle_length": 18, 
    "consumption": 201.820057840835, 
    "delay": 68, 
    "activity": "None", 
    "type": "appliance"
  }, 
  "shortFlow": {
    "cycle_length": 1, 
    "cal": 0.042562311913397724, 
    "owner": 0.0, 
    "standby_flow": 0, 
    "activity": "Presence", 
    "type": "tapping", 
    "cycle_n": 10220, 
    "cycle_flow": 1
  }, 
  "TV1": {
    "frad": 0.65, 
    "cycle_power": 124, 
    "cal": 0.056168679073161205, 
    "standby_power": 3, 
    "owner": 0.977, 
    "cycle_n": 1464.26653737683, 
    "fconv": 0.35, 
    "name": "TV1", 
    "cycle_length": 73, 
    "consumption": 241.844438744159, 
    "delay": 0, 
    "activity": "tv", 
    "type": "appliance"
  }, 
  "MusicPlayer": {
    "frad": 0.5, 
    "cycle_power": 15, 
    "cal": 0.1298248939973467, 
    "standby_power": 2, 
    "owner": 0.9, 
    "cycle_n": 1212.76285809679, 
    "fconv": 0.5, 
    "name": "MusicPlayer", 
    "cycle_length": 60, 
    "consumption": 33.2859171552583, 
    "delay": 0, 
    "activity": "audio", 
    "type": "appliance"
  }, 
  "TV3": {
    "frad": 0.65, 
    "cycle_power": 124, 
    "cal": 0.06283000905624272, 
    "standby_power": 2, 
    "owner": 0.18, 
    "cycle_n": 1521.49596625218, 
    "fconv": 0.35, 
    "name": "TV3", 
    "cycle_length": 73, 
    "consumption": 243.360717924032, 
    "delay": 0, 
    "activity": "tv", 
    "type": "appliance"
  }, 
  "TV2": {
    "frad": 0.65, 
    "cycle_power": 124, 
    "cal": 0.055739364679278376, 
    "standby_power": 3, 
    "owner": 0.58, 
    "cycle_n": 1464.26653737683, 
    "fconv": 0.35, 
    "name": "TV2", 
    "cycle_length": 73, 
    "consumption": 241.844438744159, 
    "delay": 0, 
    "activity": "tv", 
    "type": "appliance"
  }, 
  "mediumFlow": {
    "cycle_length": 1, 
    "cal": 0.017810482681251995, 
    "owner": 0.0, 
    "standby_flow": 0, 
    "activity": "Presence", 
    "type": "tapping", 
    "cycle_n": 4380, 
    "cycle_flow": 6
  }, 
  "Iron": {
    "frad": 0.5, 
    "cycle_power": 1000, 
    "cal": 0.009646339097045481, 
    "standby_power": 0, 
    "owner": 0.9, 
    "cycle_n": 35.4594692107747, 
    "fconv": 0.5, 
    "name": "Iron", 
    "cycle_length": 30, 
    "consumption": 17.7297346053873, 
    "delay": 0, 
    "activity": "iron", 
    "type": "appliance"
  }, 
  "TumbleDryer": {
    "frad": 0.0, 
    "cycle_power": 2500, 
    "cal": 0.49474492492582306, 
    "standby_power": 1, 
    "owner": 0.416, 
    "cycle_n": 122.072915738278, 
    "fconv": 0.25, 
    "name": "TumbleDryer", 
    "cycle_length": 60, 
    "consumption": 313.820216429956, 
    "delay": 0, 
    "activity": "drying", 
    "type": "appliance",
    "program":TumbleDryer(P=[0.5,0.5])
  }, 
  "DishWasher": {
    "frad": 0.0, 
    "cycle_power": 1130.61224489796, 
    "cal": 0.04603964569480809, 
    "standby_power": 0, 
    "owner": 0.335, 
    "cycle_n": 241.476395726831, 
    "fconv": 0.25, 
    "name": "DishWasher", 
    "cycle_length": 60, 
    "consumption": 273.01616986258, 
    "delay": 0, 
    "activity": "dishes", 
    "type": "appliance"
  }, 
  "TVReceiver": {
    "frad": 0.92, 
    "cycle_power": 26.8237423726735, 
    "cal": 0.05635113292276207, 
    "standby_power": 15, 
    "owner": 0.934, 
    "cycle_n": 1464.26653737683, 
    "fconv": 0.08, 
    "name": "TVReceiver", 
    "cycle_length": 73, 
    "consumption": 152.464284201826, 
    "delay": 0, 
    "activity": "tv", 
    "type": "appliance"
  }, 
  "showerFlow": {
    "cycle_length": 5, 
    "cal": 0.0204493968188275, 
    "owner": 0.0, 
    "standby_flow": 0, 
    "activity": "shower", 
    "type": "tapping", 
    "cycle_n": 73, 
    "cycle_flow": 8
  }, 
  "WashingMachine": {
    "frad": 0.0, 
    "cycle_power": 405.54347826087, 
    "cal": 0.2632287603579244, 
    "standby_power": 1, 
    "owner": 0.781, 
    "cycle_n": 195.906595865545, 
    "fconv": 0.25, 
    "name": "WashingMachine", 
    "cycle_length": 138, 
    "consumption": 191.041292123096, 
    "delay": 0, 
    "activity": "washing", 
    "type": "appliance",
    "program":WashingMachine(P=[0.5, 0.5])
  }, 
  "Oven": {
    "frad": 0.17, 
    "cycle_power": 2125, 
    "cal": 0.019262659189245385, 
    "standby_power": 3, 
    "owner": 0.616, 
    "cycle_n": 219.792801008503, 
    "fconv": 0.83, 
    "name": "Hob", 
    "cycle_length": 27, 
    "consumption": 236.16014568302, 
    "delay": 0, 
    "activity": "food", 
    "type": "appliance"
  }, 
  "AnswerMachine": {
    "frad": 0.5, 
    "cycle_power": 1, 
    "cal": 4.14823389814581e-11, 
    "standby_power": 1, 
    "owner": 0.9, 
    "cycle_n": 1e-05, 
    "fconv": 0.5, 
    "name": "AnswerMachine", 
    "cycle_length": 0, 
    "consumption": 8.76, 
    "delay": 0, 
    "activity": "None", 
    "type": "appliance"
  }
}