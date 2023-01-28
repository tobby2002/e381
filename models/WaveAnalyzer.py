from models.MonoWave import MonoWaveUp, MonoWaveDown
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.WaveCycle import WaveCycle
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, Correction, TDWave
import numpy as np
import pandas as pd
from tapy import Indicators

def fractals_only_hi_lo(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_high', 'fractals_low']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

class WaveAnalyzer:
    """
    Find impulse or corrective waves for given dataframe
    """
    def __init__(self,
                 df: pd.DataFrame,
                 verbose: bool = False):


        self.df = df
        self.lows = np.array(list(self.df['Low']))
        self.highs = np.array(list(self.df['High']))
        self.dates = np.array(list(self.df['Date']))

        # df = fractals_only_hi_lo(df)
        # self.lows = np.array(list(df['Low']))
        # self.highs = np.array(list(df['High']))
        # self.dates = np.array(list(df['Date']))


        self.verbose = verbose

        self.impulse_rules = list()
        self.correction_rules = list()

        self.__waveoptions_up: WaveOptionsGenerator5
        self.__waveoptions_down: WaveOptionsGenerator3

        self.set_combinatorial_limits()

    def get_absolute_low(self):
        """
        find the absolute low in the dataframe. Can be used to start the wave analysis from this low.
        :return:
        """
        return np.min(self.lows)

    def set_combinatorial_limits(self, n_up: int = 10, n_down: int = 10):
        """
        Change the limit to skip min / maxima for the WaveOptionsGenerators, e.g. go up to [n_up, n_up, ...] for the
        WaveOptions

        :param n_up:
        :param n_down:
        :return:
        """
        self.__waveoptions_up = WaveOptionsGenerator5(n_up)
        self.__waveoptions_down = WaveOptionsGenerator3(n_down)

    def find_impulsive_wave(self,
                            idx_start: int,
                            wave_config: list = None):
        """
        Tries to find 5 consecutive waves (up, down, up, down, up) to build an impulsive 12345 wave

        :param idx_start: index in dataframe to start from
        :param wave_config: WaveOptions
        :return: list of the 5 MonoWaves in case they are found.

                False otherwise
        """

        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]

        wave1 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        wave1.label = '1'
        wave1_end = wave1.idx_end
        if wave1_end is None:
            # if self.verbose: print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = '2'
        wave2_end = wave2.idx_end
        if wave2_end is None:
            # if self.verbose: print("Wave 2 has no End in Data")
            return False

        try:
            if wave1.high_idx == wave2.low_idx:
                return False
            if wave1.high < np.max(self.highs[wave2.high_idx:wave2.low_idx]):
                return False
            if wave1.low > np.min(self.lows[wave2.high_idx:wave2.low_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave1.low_idx ~ < wave2.high_idx : %s' % e)
            return False

        wave3 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave2_end, skip=wave_config[2])
        wave3.label = '3'
        wave3_end = wave3.idx_end
        if wave3_end is None:
            # if self.verbose: print("Wave 3 has no End in Data")
            return False

        wave4 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave3_end, skip=wave_config[3])
        wave4.label = '4'
        wave4_end = wave4.idx_end

        if wave4_end is None:
            # if self.verbose: print("Wave 4 has no End in Data")
            return False

        try:
            if wave2.low_idx == wave4.low_idx:
                return False
            if wave2.low > np.min(self.lows[wave2.low_idx:wave4.low_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave2.low > np.min')
            print(e)
            return False

        ###############
        try:
            if wave3.high_idx == wave4.low_idx:
                return False
            if wave3.high < np.max(self.highs[wave4.high_idx:wave4.low_idx]):
                return False
            if wave3.low > np.min(self.lows[wave4.high_idx:wave4.low_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave3.low_idx ~ wave4.high_idx : %s' % e)
            return False
        ###############


        wave5 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave4_end, skip=wave_config[4])
        wave5.label = '5'
        wave5_end = wave5.idx_end
        if wave5_end is None:
            # if self.verbose: print("Wave 5 has no End in Data")
            return False
        try:
            if wave4.low_idx == wave5.high_idx:
                return False
            if wave4.low > np.min(self.lows[wave4.low_idx:wave5.high_idx]):
                # if self.verbose: print('Low of Wave 4 higher than a low between Wave 4 and Wave 5')
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave4.low > np.min')
            print(e)
            return False
        # print('wave 5 ok')

        if wave1_end < wave2_end and wave2_end < wave3_end and wave3_end < wave4_end and wave4_end < wave5_end:
            pass
        else:
            return False
        if wave1.date_end < wave2.date_end and wave2.date_end < wave3.date_end and wave3.date_end < wave4.date_end and wave4.date_end < wave5.date_end:
            pass
        else:
            return False
        return [wave1, wave2, wave3, wave4, wave5]

    def find_downimpulsive_wave(self,
                            idx_start: int,
                            wave_config: list = None):
        """
        Tries to find 5 consecutive waves (down, up, down, up, down) to build a downimpulsive 12345 wave

        :param idx_start: index in dataframe to start from
        :param wave_config: WaveOptions
        :return: list of the 5 MonoWaves in case they are found.

                False otherwise
        """

        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]

        wave1 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        wave1.label = '1'
        wave1_end = wave1.idx_end
        if wave1_end is None:
            # if self.verbose: print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = '2'
        wave2_end = wave2.idx_end
        if wave2_end is None:
            # if self.verbose: print("Wave 2 has no End in Data")
            return False
        try:
            if wave1.low_idx == wave2.high_idx:
                return False
            if wave1.high < np.max(self.highs[wave2.low_idx:wave2.high_idx]):
                return False
            if wave1.low > np.min(self.lows[wave2.low_idx:wave2.high_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave1.low_idx ~ < wave2.high_idx : ' % e)
            return False

        wave3 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave2_end, skip=wave_config[2])
        wave3.label = '3'
        wave3_end = wave3.idx_end
        if wave3_end is None:
            # if self.verbose: print("Wave 3 has no End in Data")
            return False

        wave4 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave3_end, skip=wave_config[3])
        wave4.label = '4'
        wave4_end = wave4.idx_end

        if wave4_end is None:
            # if self.verbose: print("Wave 4 has no End in Data")
            return False

        try:
            if wave2.high_idx == wave4.high_idx:
                return False
            if wave2.high < np.max(self.highs[wave2.high_idx:wave4.high_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave2.high < np.max%s' % e)
            return False

        ###############
        try:
            if wave3.low_idx == wave4.high_idx:
                return False
            if wave3.high < np.max(self.highs[wave4.low_idx:wave4.high_idx]):
                return False
            if wave3.low > np.min(self.lows[wave4.low_idx:wave4.high_idx]):
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave3.low_idx ~ < wave4.high_idx : ' % e)
            return False
        ###############

        wave5 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave4_end, skip=wave_config[4])
        wave5.label = '5'
        wave5_end = wave5.idx_end
        if wave5_end is None:
            # if self.verbose: print("Wave 5 has no End in Data")
            return False
        try:
            if wave4.high_idx == wave5.low_idx:
                return False
            if wave4.high < np.max(self.highs[wave4.high_idx:wave5.low_idx]):
                # if self.verbose: print('Low of Wave 4 higher than a low between Wave 4 and Wave 5')
                return False
        except Exception as e:  # raised if `wavex` is empty.
            print('wave4.high < np.max:%s' % e)
            return False
        # print('wave 5 ok')

        if wave1_end < wave2_end and wave2_end < wave3_end and wave3_end < wave4_end and wave4_end < wave5_end:
            pass
        else:
            return False
        if wave1.date_end < wave2.date_end and wave2.date_end < wave3.date_end and wave3.date_end < wave4.date_end and wave4.date_end < wave5.date_end:
            pass
        else:
            return False
        return [wave1, wave2, wave3, wave4, wave5]

    def find_corrective_wave(self,
                             idx_start: int,
                             wave_config: list = None):
        """

        Tries to find a corrective movement (ABC)
        :param idx_start:
        :param wave_config:
        :return: a list of 3 MonoWaves (down, up, down) otherwise False

        """
        if wave_config is None:
            wave_config = [0, 0, 0]

        waveA = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        waveA.label = 'A'
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False

        waveB = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveA_end, skip=wave_config[1])
        waveB.label = 'B'
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False

        ###############
        # try:
        #     if waveB.high < np.max(self.highs[waveB.low_idx:waveB.high_idx]):
        #         return False
        #     if waveB.low > np.min(self.lows[waveB.low_idx:waveB.high_idx]):
        #         return False
        # except Exception as e:  # raised if `wavex` is empty.
        #     print('waveB.low_idx ~ < waveB.high_idx : ' % e)
        #     return False
        ###############

        waveC = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveB_end, skip=wave_config[2])
        waveC.label = 'C'
        waveC_end = waveC.idx_end
        if waveC_end is None:
            return False

        return [waveA, waveB, waveC]


    def find_upcorrective_wave(self,
                             idx_start: int,
                             wave_config: list = None):
        """

        Tries to find a corrective movement (ABC)
        :param idx_start:
        :param wave_config:
        :return: a list of 3 MonoWaves (up, down, up) otherwise False

        """
        if wave_config is None:
            wave_config = [0, 0, 0]

        waveA = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        waveA.label = 'A'
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False

        waveB = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveA_end, skip=wave_config[1])
        waveB.label = 'B'
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False

        waveC = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveB_end, skip=wave_config[2])
        waveC.label = 'C'
        waveC_end = waveC.idx_end
        if waveC_end is None:
            return False

        return [waveA, waveB, waveC]


    def find_td_wave(self, idx_start: int, wave_config: list = None):
        if wave_config is None:
            wave_config = [0, 0]

        wave1 = MonoWaveUp(self.df, idx_start=idx_start, skip=wave_config[0])
        wave1.label = '1'
        wave1_end = wave1.idx_end
        if wave1_end is None:
            if self.verbose: print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(self.df, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = '2'
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose: print("Wave 2 has no End in Data")
            return False

        return [wave1, wave2]

    def next_cycle(self,
                   start_idx: int):

        impulse = Impulse('impulse')
        correction = Correction('correction')

        wave_cycles = set()

        for new_option_impulse in self.__waveoptions_up.options_sorted:

            cycle_complete = False
            waves_up = self.find_impulsive_wave(idx_start=start_idx,
                                                wave_config=new_option_impulse.values)

            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=False)
                if wavepattern_up.check_rule(impulse):
                    if self.verbose: ('Impulse found!', new_option_impulse.values)
                    end = waves_up[4].idx_end

                    for new_option_correction in self.__waveoptions_down.options_sorted:
                        waves = self.find_corrective_wave(idx_start=end, wave_config=new_option_correction.values)
                        if waves:
                            wavepattern = WavePattern(waves, verbose=False)
                            if wavepattern.check_rule(correction):

                                cycle_complete = True
                                wave_cycle = WaveCycle(wavepattern_up, wavepattern)
                                wave_cycles.add(wave_cycle)

                                # if wave_cycle not in wave_cycles:
                                if self.verbose and wave_cycle not in wave_cycles:
                                    print('Corrrection found!', new_option_correction.values)
                                    print('*' * 40)

                    if cycle_complete:
                        yield wave_cycle

        return None
