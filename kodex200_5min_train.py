import numpy as np
import pandas as pd


class Kodex200:
    def __init__(self, seed=0):
        np.random.seed(seed)

        # observation shape
        # [close, close-open, high-low, volume, current_bar, order_remained], reward, done
        self.observation_size = 6
        self.action_size = 1

        # initial value
        self.current_index = 0
        self.current_step = 0
        self.order_remained = 1.0
        self.reward = 0.0
        self.done = False

        self.sum_volume = 0.0
        self.sum_volumeprice = 0.0
        self.vwap = 0.0

        self.sum_volume_a = 0.0
        self.sum_volumeprice_a = 0.0
        self.vwap_action = 0.0

        self.total_index, self.total_step, self.data = self._build()

    def step(self, action):
        if self.done:
            print("Warning! You are calling 'step()' even though this environment has already returned done = True."
                  "You should always call 'reset()' once you receive 'done = True'"
                  "-- any further steps are undefined behavior.")
        else:
            self.current_step += 1

        if 'numpy' in str(type(action)):
            action = action.item()
        if action > 1 or action < 0:
            raise ValueError

        self.order_remained -= action

        state = self.data[self.current_step, :, self.current_index]
        state = np.append(state, self.order_remained)
        state = np.reshape(state, [1, 1, self.observation_size])

        self.sum_volume += state[0, 0, 3]
        self.sum_volumeprice += state[0, 0, 3] * state[0, 0, 1]
        self.vwap = self.sum_volumeprice / self.sum_volume

        self.sum_volume_a += action
        self.sum_volumeprice_a = action * state[0, 0, 1]
        self.vwap_action = self.sum_volumeprice_a / self.sum_volume_a

        # reward of execution order completeness
        if self.current_step == self.total_step-1:
            self.done = True
            # reward = -10 to 10, when order_remained 1 to 0
            self.reward = 10 * (1 - 2 * self.order_remained)
        else:
            # constraint: twap +- 20%
            # if self.order_remained < 0.8 * (1 - state[0, 0, 4])\
            #         or self.order_remained > 1.2 * (1 - state[0, 0, 4]):
            if self.order_remained < 0:
                self.done = True
                self.reward = -1.0
            else:
                self.done = False
                # basic reward for continuous step, total = 10
                self.reward = 10 / (self.total_step-2)

        # reward of vwap performance
        if self.vwap_action < self.vwap:
            if self.done:
                self.reward += 10
            else:
                self.reward += 10 / (self.total_step-2)

        return state, self.reward, self.done, (self.vwap - self.vwap_action)/self.vwap

    def reset(self):
        self.current_index = np.random.randint(self.total_index)
        self.current_step = 0
        self.order_remained = 1.
        self.reward = 0.
        self.done = False

        self.sum_volume = 0.0
        self.sum_volumeprice = 0.0
        self.vwap = 0.0

        self.sum_volume_a = 0.0
        self.sum_volumeprice_a = 0.0
        self.vwap_action = 0.0

        state = self.data[self.current_step, :, self.current_index]
        state = np.append(state, self.order_remained)

        state = np.reshape(state, [1, 1, self.observation_size])

        return state

    def _build(self):
        # KODEX200 5min data 2019-12-18 ~ 2020-03-06
        # 2020-01-02: market opened 1 hour later, for analysis efficiency exclude these data
        # total 52 days
        file_path = "./kodex200_5min.csv"
        df = pd.read_csv(file_path)

        # Sort ascending by date and time
        df = df.iloc[::-1]

        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.drop(['date', 'time'], axis=1, inplace=True)

        COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[COLUMNS]

        # Market trading time = 09:00:00 ~ 15:30:00
        # But 15:20:00 ~ 15:30:00 is simultaneous quotation time. Exclude after 15:20:00
        indexnames = df[(df['datetime'].dt.time >= pd.to_datetime('15:20:00').time())].index
        df.drop(indexnames, inplace=True)

        # Trading time is 09:00:00 ~ 15:20:00
        # 60 min * 6 h + 20 min = 380 min
        tradingtime = 380
        # Total number of 5-min bar per day : totalbars = 76
        totalbars = int(tradingtime / 5)
        # Total number of days : totaldays = 52
        totaldays = int(len(df) / totalbars)

        # Check incomplete data
        # correct time range is 09:00:00 ~ 15:20:00
        # Check time of 1st bar data
        def checkdata(data, nbars):
            for i in range(int(len(data) / nbars)):
                starttime = pd.to_datetime('9:00:00').time()
                if data.iloc[i * nbars]['datetime'].time() != starttime:
                    print(i * nbars, data.iloc[i * nbars][['datetime']])
                    break

        checkdata(df, totalbars)

        # re-indexing data, ascending order by date
        df.index = range(len(df))

        # modify and rearrange columns
        # diff = close - open
        # highlow = high - low
        close = pd.Series(df['close'], name='close')
        diff = pd.Series((df['close'] - df['open']), name='diff')
        highlow = pd.Series((df['high'] - df['low']), name='highlow')
        volume = pd.Series(df['volume'], name='volume')

        temp = pd.concat([close, diff, highlow, volume], axis=1)
        datalist = temp.to_numpy(dtype=np.float32)
        timelist = df['datetime'].to_numpy(dtype='datetime64[s]')

        dataset = np.empty([totalbars, 4, totaldays], dtype=np.float32)
        timeset = np.empty([totalbars, totaldays], dtype='datetime64[s]')

        # reshape
        # dataset's shape is (totalbars, features, totaldays)
        # timeset's shape is (totalbars, totaldays)
        for i in range(totaldays):
            dataset[:, :, i] = datalist[i * totalbars: (i + 1) * totalbars, :]
            timeset[:, i] = timelist[i * totalbars: (i + 1) * totalbars]

        # Normalization daily basis
        for i in range(totaldays):
            base_price = dataset[0, 0, i]
            base_volume = dataset[0, 3, i]
            for j in range(totalbars):
                dataset[j, 0, i] /= base_price
                dataset[j, 1, i] /= base_price
                dataset[j, 2, i] /= base_price
                dataset[j, 3, i] /= base_volume

        # Add timestamp column and
        # i-th timestamp is (i / totalbars)
        trainset = np.empty([totalbars, 5, totaldays], dtype=np.float32)
        for i in range(totaldays):
            bardata = np.empty([totalbars], dtype=np.float32)
            for j in range(totalbars):
                bardata[j] = (j + 1) / totalbars
            trainset[:, :, i] = np.c_[dataset[:, :, i], bardata]
        return totaldays, totalbars, trainset


if __name__ == "__main__":
    test = Kodex200()
    test.reset()

    for i in range(test.total_step):
        test_observation = test.step(0.3)
        print('step: {}     /   {}'.format(i, test_observation))
