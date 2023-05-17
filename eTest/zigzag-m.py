
def zigzag(_ohlc_df: pd.DataFrame, _depth: int, _deviation: float, _paper_fees: float) -> (list, list, float):
    """paper_fees
    Basic implementation of ZigZag indicator for pd.DataFrame python processing.

    :type _ohlc_df: pd.Dataframe
    :param _ohlc_df: dataset with olhc data of the timeseries
    :type _depth: int
    :param _depth: the usual "lenght" or "number of legs" param, it defines the mean desired number of candles in the trends
    :type _deviation: float
    :param _deviation: the price deviation for reversals (e.g. 5.0, default)

    :return filtered_pivot_indexes: time index for the calculated pivot points (x value)
    :return filtered_pivot_values: respective calulated values (y value)
    :return _roi_calculations: estimation of the total theorical profit for theorical trades using the calculated pivots
    """

    # dataset split into lists
    _high_data = _ohlc_df['High']
    _high_data_list = _high_data.tolist()
    _low_data = _ohlc_df['Low']
    _low_data_list = _low_data.tolist()

    # converting the deviation from percent to decimal
    _deviation = _deviation / 100

    # looking for high indexes through peak analysis
    _high_indices, _ = find_peaks(_high_data.tolist(), distance=_depth)

    # looking for low indexes through peak analysis
    _low_indices, _ = find_peaks([_vl * -1 for _vl in _low_data.tolist()], distance=_depth)

    # loop variable controls
    filtered_pivot_indexes = []
    filtered_pivot_values = []

    # appeding pivots and sorting (time index)
    _all_indexes = _high_indices.tolist() + _low_indices.tolist()
    _all_indexes = sorted(_all_indexes)

    # filtering by consecutives peaks and valleys order
    _last_was_a_peak = False  # to control the kind of the last added point
    for _index in _all_indexes:

        # case for the first to be added
        if not filtered_pivot_indexes:

            # appending first point
            filtered_pivot_indexes.append(_index)

            # first point being a peak
            if _high_indices[0] < _low_indices[0]:
                _last_was_a_peak = True
                filtered_pivot_values.append(_high_data_list[_index])

            # first point being a valley
            else:
                filtered_pivot_values.append(_low_data_list[_index])

            # skipping for the next loop
            continue

        # trigger control
        _t1 = _index in _high_indices
        _t2 = _index in _low_indices
        _t3 = _t1 and _last_was_a_peak
        _t4 = _t2 and not _last_was_a_peak

        # suppresing consecutive peaks
        if _t3 or _t4:

            # analysis for consecutive valleys
            if _last_was_a_peak:
                _last_added_point_value = filtered_pivot_values[-1]
                _current_point_value = _high_data_list[_index]

                # suppressing the last added valley for a lower valley level
                if _current_point_value >= _last_added_point_value:

                    # removing the last added points
                    del filtered_pivot_indexes[-1]
                    del filtered_pivot_values[-1]

                    # updating the new one
                    filtered_pivot_indexes.append(_index)
                    filtered_pivot_values.append(_high_data_list[_index])
                else:
                    continue

            # analysis for consecutive peaks
            else:
                _last_added_point_value = filtered_pivot_values[-1]
                _current_point_value = _low_data_list[_index]

                # suppressing the last added valley for a lower valley level
                if _current_point_value <= _last_added_point_value:

                    # removing the last added points
                    del filtered_pivot_indexes[-1]
                    del filtered_pivot_values[-1]

                    # updating the new one
                    filtered_pivot_indexes.append(_index)
                    filtered_pivot_values.append(_low_data_list[_index])
                else:
                    continue

        # case for the last point added was a peak
        elif _t2 and _last_was_a_peak:
            _last_was_a_peak = False
            filtered_pivot_indexes.append(_index)
            filtered_pivot_values.append(_low_data_list[_index])

        # case for the last point added was a valley
        elif _t1 and not _last_was_a_peak:
            _last_was_a_peak = True
            filtered_pivot_indexes.append(_index)
            filtered_pivot_values.append(_high_data_list[_index])

    # deviation filtering
    _total_deviation = (max(_high_data_list) - min(_low_data_list)) / min(_low_data_list)
    _minimal_deviation = abs(_total_deviation * _deviation)

    # filtering by the minimal deviation criteria
    for _index in range(len(filtered_pivot_values) - 1, 1, -1):
        try:
            _first_value = filtered_pivot_values[_index]
            _second_value = filtered_pivot_values[_index - 1]
            _variation = abs((_first_value - _second_value) / _first_value)

        # case for the remove of the last two points
        except IndexError:
            continue

        # case for not reaching the minimal deviation
        if _variation < _minimal_deviation:
            del filtered_pivot_values[_index]
            del filtered_pivot_indexes[_index]

            # alteration to keep the last pivot point
            if _index != len(filtered_pivot_values) - 1:
                del filtered_pivot_values[_index - 1]
                del filtered_pivot_indexes[_index - 1]

    # calculation of the ROI (return over investiment) parameter
    # it calculates profit for theorical buy and sell in the calculated period
    _roi_calculations = []
    for _index in range(1, len(filtered_pivot_values) - 1):
        _first_value = filtered_pivot_values[_index - 1]
        _second_value = filtered_pivot_values[_index]
        _current_roi = abs((_second_value - _first_value) / _first_value) - _paper_fees
        _roi_calculations.append(_current_roi)

    return filtered_pivot_indexes, filtered_pivot_values, sum(_roi_calculations)

# i = [3, 5, 9, 13, 16, 19, 25, 34, 37, 43, 45, 64]
# v = [0.00839, 0.00883, 0.00856, 0.00917, 0.00867, 0.00923, 0.00855, 0.00858, 0.00812, 0.00874, 0.00822, 0.00914]

def f_mw_pattern(i, v, c, n=5):
    if i is not None and len(i) >= 5:
        idxs = i[-n:]
        vals = v[-n:]
        maps = [(x, y) for x, y in zip(idxs, vals)]
        ranks = [sorted(vals, reverse=False).index(i) for i in vals]
        dv = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)] + [c[0] - idxs[-1]]
        mw = 0 if vals[0] < vals[1] else 1

        highest = max(vals)
        lowest = min(vals)
        avg = sum(vals) / len(vals)
        mid = (highest + lowest) / 2
        height = highest - lowest
        std = (avg - mid) / height
        current = c[1]
        position = (highest - current)/height if c[1] >= mid else (current - lowest)/height
        f = [mw] + ranks + dv + [std] + [position]
    else:
        return None, None
    return f, maps

# c = [70, 0.00899]
# f, maps = f_mw_pattern(i, v, c, n=5)
