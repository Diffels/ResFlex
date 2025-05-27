import os
import sys
import random
import numpy as np
from .residential import Household, Equipment
from . import stats
from . import data
from .Data.Appliances import set_appliances
from .Data.Households import households
import itertools
import pandas as pd
special_appliances = ['DishWasher','WasherDryer','TumbleDryer','WashingMachine', 'EVCharging']
StaticLoad = [x for x in set_appliances if (set_appliances[x]['type'] == 'appliance' and x not in special_appliances)]

class Household_mod(Household):  
    def parameterize(self, **kwargs):
        '''
        Get a household definition for occupants and present appliances based
        on average statistics or the given kwargs.
        '''

        def members(**kwargs):
            '''
            Define the employment type of all household members based on time
            use survey statistics or the given kwargs.
            '''
            members = []
            # First we check if membertypes are given as **kwargs
            if 'members' in kwargs:
                if isinstance(kwargs['members'], list):
                    members = kwargs['members']
                    if 'Random' in members : 
                        other_members = [x for x in members if x != 'Random']
                        if len(other_members) != 0:
                            def contain_all_values(liste, valeurs):
                                return all(valeur in liste for valeur in valeurs)
                            list_dwel = [value for value in households.values() if contain_all_values(value, other_members) and len(value) == len(members)]
                            members = random.choice(list_dwel)
                        else : 
                            list_dwel = [valeur for valeur in households.values() if len(valeur) == len(members)]
                            members = random.choice(list_dwel)
                else:
                    raise TypeError('Given membertypes is no List of strings.')
            elif 'nb_people' in kwargs :
                list_dwel = [valeur for valeur in households.values() if len(valeur) == kwargs['nb_people']]
                members = random.choice(list_dwel)
            # If no types are given, random statististics are applied
            else:
                key = random.randint(1, len(households))
                members = households[key]
            # And return the members as list fo strings
            return members

        def appliances(**kwargs):
            '''
            Define the pressent household appliances based on average national
            statistics independent of household member composition.
            '''
            # Loop through all appliances and pick randomly based on the
            # rate of ownership.
            # changes for new cold-appliance fix #######################################
            
            # Based on 10000 runs, these new values combined with rule-based fix below
            # lead to the same overall ownership as the original values.
            # We change it here so that the original remain in the Appliances file.
            set_appliances['Refrigerator']['owner']=0.27     # original:  0.430
            set_appliances['FridgeFreezer']['owner']=0.40    # original:  0.651
            set_appliances['ChestFreezer']['owner']=0.19     # original:  0.163
            set_appliances['UprightFreezer']['owner']=0.31   # original:  0.291
            
            app_n = []
            for app in set_appliances:
                if set_appliances[app]['type'] == 'appliance':
                    obj = Equipment(**set_appliances[app])
                    if 'selected_appliances' in kwargs:
                        if set_appliances[app]['name'] in kwargs['selected_appliances'] :
                            owner = float(kwargs['selected_appliances'][app]) >= random.random()
                        elif set_appliances[app]['name'] not in special_appliances : 
                            owner = obj.owner >= random.random()
                    else :
                        owner = obj.owner >= random.random()   
                    app_n.append(app) if owner else None
                    
            # Cold appliances fix:   ###############################################        
            if not ('FridgeFreezer' in app_n) and not ('Refrigerator' in app_n): # if there was none of the two-> add one of the two.
                #  Find probability of household to own FF instead of R: (FF ownership over sum of two ownerships-> scale to 0-1 interval)
                prob=set_appliances['FridgeFreezer']['owner']/(set_appliances['FridgeFreezer']['owner']+set_appliances['Refrigerator']['owner'])
                # if random number is below prob, then the household will own a FF, otherwise a R -> add it
                app_n.append('FridgeFreezer') if prob >= random.random()  else app_n.append('Refrigerator') 
            
            if 'FridgeFreezer' in app_n and 'ChestFreezer' in app_n and 'UprightFreezer' in app_n:  #if there were 3 freezers-> remove a freezer-only
                #find probability of household to own CF instead of UF:  (CF ownership over sum of two ownerships-> scale to 0-1 interval)
                prob=set_appliances['ChestFreezer']['owner']/(set_appliances['ChestFreezer']['owner']+set_appliances['UprightFreezer']['owner'])
                # if random number is below prob, then the household will own a CF, otherwise an UF-> remove the other
                app_n.remove('UprightFreezer') if prob >= random.random()  else app_n.remove('ChestFreezer') #remove the one you don't own
                
            #########################################################################
            return app_n

        def tappings():
            '''
            Define the present household tapping types.
            '''
            tap_n = ['shortFlow', 'mediumFlow', 'bathFlow', 'showerFlow']
            return tap_n

        def clusters(members):
            '''
            Allocate each household member to the correct cluster based on the
            members occupation in time use survey data.
            '''
            clustersList = []
            # loop for every individual in the household
            for ind in members:
                if ind != 'U12':
                    clu_i = data.get_clusters(ind)
                    clustersList.append(clu_i)
            # and return the list of clusters
            return clustersList
        # and run all
        self.members = members(**kwargs)
        self.apps = appliances(**kwargs)
        self.taps = tappings()
        self.clustersList = clusters(self.members)
        # and return
        # print('Household-object created and parameterized.')
        # print(' - Employment types are %s' % str(self.members))
        summary = [] #loop dics and remove doubles
        for member in self.clustersList:
            summary += member.values()
        # print(' - Set of clusters is %s' % str(list(set(summary))))

        return None
      
    def __shsetting__(self):
        '''
        Simulation of the space heating setting points.
        '''

        #######################################################################
        # we define setting types based on their setpoint temperatures 
        # when being active (1), sleeping (2) or absent (3).
        types = dict()
        types.update({'2' : {1:18.5, 2:15.0, 3:18.5}})
        types.update({'3' : {1:20.0, 2:15.0, 3:19.5}})
        types.update({'4' : {1:20.0, 2:11.0, 3:19.5}})
        types.update({'5' : {1:20.0, 2:14.5, 3:15.0}})
        types.update({'6' : {1:21.0, 2:20.5, 3:21.0}})
        types.update({'7' : {1:21.5, 2:15.5, 3:21.5}})
        # and the probabilities these types occur based on Dutch research,
        # i.e. Leidelmeijer and van Grieken (2005).
        types.update({'prob' : [0.16, 0.35, 0.08, 0.11, 0.05, 0.20]})
        # and given the type, denote which rooms are heated (more than one possibility)
        shr = dict()
        shr.update({'2' : [['dayzone','bathroom']]})
        shr.update({'3' : [['dayzone'],['dayzone','bathroom'],['dayzone','nightzone']]})
        shr.update({'4' : [['dayzone'],['dayzone','nightzone']]})
        shr.update({'5' : [['dayzone']]})
        shr.update({'6' : [['dayzone','bathroom','nightzone']]})
        shr.update({'7' : [['dayzone','bathroom']]})
    
        #######################################################################
        # select a type based on random number and probabilities associated to types
        rnd = np.random.random()
        shtype = str(1 + stats.get_probability(rnd, types['prob'], 'prob'))
        #define which rooms will be heated
        if len(shr[shtype]) != 1: # if there are more possibilities, choose one randomly
            nr = np.random.randint(len(shr[shtype]))
            shrooms = shr[shtype][nr]
        else:
            shrooms = shr[shtype][0]
    
        #######################################################################
        # create a profile for the heated rooms
        shnon = 12*np.ones(len(self.occ_m[0])) #non-heated rooms : 12 degC
        shset = 12*np.ones(len(self.occ_m[0]))  #initiate space heating settings also as non-heated
        occu = self.occ_m[0] # get merged occupancy 
        for key in types[shtype].keys(): # for each occupancy state
            shset[occu == key] = types[shtype][key]  # use appropriate temperature setting given in "types"   

        #######################################################################
        # and couple to the heated rooms
        sh_settings = dict()
        for room in ['dayzone', 'nightzone', 'bathroom']:
            if room in shrooms:
                sh_settings.update({room:shset})
            else:
                sh_settings.update({room:shnon})
        # and store
        self.sh_settings = sh_settings
        # print(' - Average comfort setting is %s Celsius' % str(round(np.average(sh_settings['dayzone']),2)))
        self.variables.update({'sh_day': 'Space heating set-point temperature for day-zone in degrees Celsius.',
                                'sh_bath': 'Space heating set-point temperature for bathroom in degrees Celsius.',
                                'sh_night': 'Space heating set-point temperature for night-zone in degrees Celsius.'})
     
        return None
    def __plugload__(self):
        '''
        Simulation of the electric load based on the occupancy profile, 
        the clusters that determine the activity probabilities,
        and the present appliances.
        - Weekdays, Saturday and Sunday differ.
        - The simulation starts at 4:00 AM.
        '''

        def receptacles(self):
            '''
            Simulation of the receptacle loads.
            '''
            nday = self.nday
            dow = self.dow
            nmin = nmin = nday * 24 * 60 # number of minutes in total number of days
            power = np.zeros(nmin+1)
            react = np.zeros(nmin+1)
            radi = np.zeros(nmin+1)
            conv = np.zeros(nmin+1)
            result_n = dict()
            app_consumption = dict()
            for app in self.apps:
                # create the equipment object with data from Appliances.py
                eq = Equipment(**set_appliances[app])
                # simulate what the load will be
                r_app, n_app = eq.simulate(nday, dow, self.clustersList, self.occ)
                # and add to total load
                result_n.update({app:n_app})
                power += r_app['P']
                react += r_app['Q']
                app_consumption.update({app : r_app['P']})
                radi += r_app['QRad']
                conv += r_app['QCon']

            self.app_consumption = app_consumption
            # a new time axis for power output is to be created 
            # since a different time step is used in comparison to occupancy
            time = 4*60*60 + np.arange(0, (nmin+1)*60, 60) # in seconds, starts at 4:00 AM

            result = {'time':time, 'P':power, 'Q':react, 'QRad':radi, 'QCon':conv }
        
            self.r_receptacles = result
            self.n_receptacles = result_n
            return None

        def lightingload(self):
            '''
            Simulate use of lighting for residential buildings based on the
            model of J. Widén et al (2009)
            The model computes an ideal power level pow_id depending on the irradiance level 
            and occupancy (but not on light switching behavior of occupants itself). 
            The power is then adjusted in steps, assuming occupants will not react smoothly to irradiation changes.
            '''

            # parameters ######################################################
            # Simulation of lighting load requires information on irradiance
            # levels which determine the need for lighting if occupant.
            # The loaded solar data represent the global horizontal radiation
            # at a time-step of 1-minute for Uccle, Belgium
            # Since the data includes only 365 days, we add one days at the beginning to match the extra initiation day
            # and one at the end in case of a leap year.
            # Furthermore, the data starts at midnight, so a shift to 4am is necessary
            # so that it coincides with the occupancy data!!! (the first 4 h are moved to the end) 
            irr = np.loadtxt(os.path.join(os.path.dirname(__file__), 'Data', 'Climate', 'irradiance.txt'), float)
            irr=np.insert(irr,1,irr[-24*60:]) # add december 31 to start of year (for extra day used to fill first 4h)
            irr=np.append(irr,irr[-24*60:]) # add december 31 to end of year in case of leap year
            irr = np.roll(irr,-240) # brings first 4h to end, to match start of occupancy at 4 AM instead of midnight
            # script ##########################################################
            # a yearly simulation is basic, also in a unittest
            nday = self.nday
            #nbin = 144 # steps in occupancy data per day (10min steps)
            nmin = nday * 24 * 60 # number of minutes in total number of days
            occ_m = self.occ_m[0]
            to = -1 # time counter for occupancy
            tl = -1 # time counter for minutes in lighting load
            power_max = 200/4 # lighting power used when there is 0 irradiance (maximum)
            irr_max = 200 # irradiance threshhold above which no lighting is used
            pow_id = np.zeros(nmin+1) # initialize zero ideal lighing load
            prob_adj = 0.1 # probability to adjust lighting
            pow_adj = 40 # step by which power is adjusted
            P = np.zeros(nmin+1)
            Q = np.zeros(nmin+1)
            for m in range(len(occ_m)): #loop over simulation period, per 1min steps
                to += 1
  #              for run in range(0, 10): #for each minute in the step
                tl += 1
                if occ_m[m] > int(1) or (irr[m] >= irr_max): #if occupants not active (occ_m>1) OR irradiance more than threshhold
                    pow_id[m] = 0 # lights will be off
                else:
                    pow_id[m] = power_max*(1 - irr[m]/irr_max) # lights ON, power depends on level of irradiance compared to irr_max
                
                # determine final power usage after stepwise adjustments
                if occ_m[to] > int(1): # if OFF, it stays that way
                    P[m] = pow_id[m]
                elif random.random() <= prob_adj: # if ON, check if adjustment happens (random number< prob_adj)
                    delta = P[m-1] - pow_id[m] # difference between previous step and "ideal" current step 
                    if delta > 0 and pow_adj/2 < np.abs(delta) : # if absolute difference is larger than half of the adjustment step
                        P[m] = P[m-1]-pow_adj  # the new power is the previous one, decreased by the adjustment step
                    elif delta < 0 and pow_adj/2 < np.abs(delta):
                        P[m] = P[m-1]+pow_adj  # the new power is the previous one, increased by the adjustment step
                    else: #otherwise, keep previous level
                        P[m] = P[m-1]
                else: #otherwise, keep previous level
                    P[m] = P[m-1]

            radi = P*0.55 # fixed radiative part 55% for heat dissipation
            conv = P*0.45 # fixed convective part 45% for heat dissipation

            result = {'P':P, 'Q':Q, 'QRad':radi, 'QCon':conv}

            self.r_lighting = result
            
            return None

        receptacles(self)
        lightingload(self)
 
        self.variables.update({'P': 'Active power demand for appliances and lighting in W.',
                               'Q':'Reactive power demand for appliances and lighting in W.',
                               'QRad': 'Radiative internal heat gains from appliances and lighting in W.',
                               'QCon': 'Convective internal heat gains from appliances and lighting in W.'})

        return None
    def roundUp(self):
        '''
        Round the simulation by wrapping all data and reduce storage size.
        Also shift the data to start at Midnight instead of 4 AM (last 4h brought to the front).
        Keeps all data in 1-minute resolution.
        '''

        #######################################################################
        # first we move and summarize data to the most upper level.
        self.sh_day = self.sh_settings['dayzone']
        self.sh_night = self.sh_settings['nightzone']
        self.sh_bath = self.sh_settings['bathroom']
        self.P = self.r_receptacles['P'] + self.r_lighting['P']
        self.Q = self.r_receptacles['Q'] + self.r_lighting['Q']
        self.QRad = self.r_receptacles['QRad'] + self.r_lighting['QRad']
        self.QCon = self.r_receptacles['QCon'] + self.r_lighting['QCon']
        self.mDHW = self.r_flows['mDHW']
        self.dow = self.dow[1]  # only first day of year is interesting to keep (skip initiation day(index=0))
        
        #######################################################################        
        # delete first 20h and last 4h so that data starts and ends at midnight
        # keep all data in 1-minute resolution
        self.nday = self.nday - 1  # change back to originally asked number (remove extra initiation day)
        start = 20 * 60  # start minute, after 20h -> midnight of initiation day
        stop = start + self.nday * 24 * 60  # end minute, 4h before end of last day -> midnight 
        self.occ_m = self.occ_m[0][start:stop + 1]  # 1-min resolution
        self.occ = [i[start:stop + 1] for i in self.occ]
    
        self.sh_day = self.sh_day[start:stop + 1]
        self.sh_night = self.sh_night[start:stop + 1]
        self.sh_bath = self.sh_bath[start:stop + 1]
        self.P = self.P[start:stop + 1]  # 1-min data, and one extra step
        self.Q = self.Q[start:stop + 1]
        self.QRad = self.QRad[start:stop + 1]
        self.QCon = self.QCon[start:stop + 1]
        self.mDHW = self.mDHW[start:stop + 1]
        appliances_cons = self.app_consumption
        
        new_appliance = dict()
        for key, value in appliances_cons.items():
            new_appliance.update({key: value[start:stop + 1]})
        del self.app_consumption
        df_P = pd.DataFrame(new_appliance)
        StaticLoad_pres = [col for col in StaticLoad if col in df_P.columns]
        df_P.loc[:, 'BaseLoad'] = df_P[StaticLoad_pres].sum(axis=1)
        df_P = df_P.drop(columns=StaticLoad_pres)
        df_P = df_P.iloc[:-1]

        #app = pd.DataFrame(new_appliance)
        self.app_consumption = df_P
        
        #######################################################################
        # then we delete the old data structure to save space
        del self.sh_settings
        del self.r_receptacles
        del self.r_lighting
        del self.r_flows
        #######################################################################
        # and end
        return None

# Create and simulate a single household, with given type of members, and given year
