"""
Different negative sampling strategy

Date:
    - Aug. 14, 2022
"""
import numpy as np
import torch


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'  # negative edge sampling method: random edges
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class RandEdgeSampler_NE(object):
    """
    Random Edge Sampling for Negative Edge Selection
    Different Strategies are available for selecting the negative edges
    """

    def __init__(self, src_list, dst_list, ts_list, last_ts_train_val, NS='rnd', seed=None, rnd_sample_ratio=1.0):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """

        self.seed = None
        self.neg_sample = NS  # Negative Sampling Strategy
        self.rnd_sample_ratio = rnd_sample_ratio
        if self.rnd_sample_ratio == 1.0:
            self.neg_sample = 'rnd'
            print("INFO: Setting rnd_sample_ratio = 1.0 is equivalent to using Random Sampling Strategy.")
        self.src_list = src_list
        self.dst_list = dst_list
        self.ts_list = ts_list
        self.src_list_distinct = np.unique(src_list)
        self.dst_list_distinct = np.unique(dst_list)
        self.ts_list_distinct = np.unique(ts_list)
        self.ts_init = min(self.ts_list_distinct)
        self.ts_end = max(self.ts_list_distinct)
        self.ts_last_before_test = last_ts_train_val
        self.e_train_val_l = self.get_edges_in_time_interval(self.ts_init, self.ts_last_before_test)
        self.replace = False

        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
            self.random_state = np.random.RandomState(self.seed)

    def get_edges_in_time_interval(self, start_ts, end_ts):
        """
        return positive edges of a specific time interval as a dictionary
        """
        valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
        interval_src_l = self.src_list[valid_ts_interval]
        interval_dst_l = self.dst_list[valid_ts_interval]
        interval_edges = {}
        for src, dst in zip(interval_src_l, interval_dst_l):
            if (src, dst) not in interval_edges:
                interval_edges[(src, dst)] = 1
        return interval_edges

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, size, current_split_start_ts, current_split_end_ts):
        if self.neg_sample == 'hist':
            neg_hard_e_source, neg_hard_e_dest, neg_rnd_source, neg_rnd_dest = self.sample_hist_NE(size,
                                                                                                   current_split_start_ts,
                                                                                                   current_split_end_ts)
        elif self.neg_sample == 'induc':
            neg_hard_e_source, neg_hard_e_dest, neg_rnd_source, neg_rnd_dest = self.sample_induc_NE(size,
                                                                                                    current_split_start_ts,
                                                                                                    current_split_end_ts)
        elif self.neg_sample == 'rnd':
            neg_hard_e_source, neg_hard_e_dest, neg_rnd_source, neg_rnd_dest = self.sample_rnd_NE(size,
                                                                                                  current_split_start_ts,
                                                                                                  current_split_end_ts)
        else:
            raise ValueError("INFO: {}:  Undefined Negative Edge Sampling Strategy!".format(self.neg_sample))
        return neg_hard_e_source, neg_hard_e_dest, neg_rnd_source, neg_rnd_dest

    def sample_hist_NE(self, size, current_split_start_ts, current_split_end_ts):
        """
        sample negative edges based on "Historical Negative Sampling Strategy"
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        # Note: if self.rnd_sample_ratio != 1, historical edges cannot be positive at the same time!
        non_repeating_e_src_l, non_repeating_e_dst_l = self.get_difference_edge_list(history_e_dict,
                                                                                     current_split_e_dict)
        num_smp_rnd = int(size * self.rnd_sample_ratio)
        num_smp_from_hist = size - num_smp_rnd
        if num_smp_from_hist > len(non_repeating_e_src_l):
            num_smp_from_hist = len(non_repeating_e_src_l)
            num_smp_rnd = size - num_smp_from_hist

        # random negative edges
        neg_rnd_source, neg_rnd_dest = self._get_random_sample(num_smp_rnd, current_split_e_dict, history_e_dict)

        # negative historical edges
        nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=self.replace)
        neg_hist_source = non_repeating_e_src_l[nre_e_index]
        neg_hist_dest = non_repeating_e_dst_l[nre_e_index]

        return neg_hist_source, neg_hist_dest, neg_rnd_source, neg_rnd_dest

    def sample_induc_NE(self, size, current_split_start_ts, current_split_end_ts):
        """
        sample negative edges based on "Inductive Negative Sampling Strategy"
        NOTE:
          1. This should only be used for evaluation (not training)!
          2. All edges are selected according to the standard setting, unless we don't have enough inductive negative edges
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
        curr_split_induc_negative_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)

        neg_induc_source, neg_induc_dst = [], []
        if len(curr_split_induc_negative_e) > 0:
            for e in curr_split_induc_negative_e:
                neg_induc_source.append(e[0])
                neg_induc_dst.append(e[1])
            neg_induc_source = np.array(neg_induc_source)
            neg_induc_dst = np.array(neg_induc_dst)
        num_smp_rnd = size - len(curr_split_induc_negative_e)

        if num_smp_rnd > 0:
            # random negative edges
            neg_rnd_source, neg_rnd_dest = self._get_random_sample(size, current_split_e_dict, history_e_dict)
            return neg_induc_source, neg_induc_dst, neg_rnd_source, neg_rnd_dest
        else:
            rnd_induc_hist_index = np.random.choice(len(curr_split_induc_negative_e), size=size, replace=self.replace)
            neg_dummy_source, neg_dummy_dest = [], []
            return neg_induc_source[rnd_induc_hist_index], neg_induc_dst[
                rnd_induc_hist_index], neg_dummy_source, neg_dummy_dest

    def sample_rnd_NE(self, size, current_split_start_ts, current_split_end_ts):
        """
        sample negative edges based on "Random Negative Sampling Strategy"
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        current_split_e_dict = self.get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)

        # random negative edges
        neg_rnd_source, neg_rnd_dest = self._get_random_sample(size, current_split_e_dict, history_e_dict)
        neg_dummy_source, neg_dummy_dest = [], []
        return neg_dummy_source, neg_dummy_dest, neg_rnd_source, neg_rnd_dest

    def _get_random_sample(self, size, current_split_e_dict, history_e_dict):
        num_selected_smp_rnd = 0
        neg_rnd_source, neg_rnd_dest = [], []
        while num_selected_smp_rnd < size:
            current_rnd_src_index = np.random.choice(len(self.src_list_distinct), size=1, replace=self.replace)
            current_rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=1, replace=self.replace)
            selected_rnd_src = self.src_list_distinct[current_rnd_src_index][0]
            selected_rnd_dst = self.dst_list_distinct[current_rnd_dst_index][0]
            # accept/reject mechanism: to make sure that the selected edge is not a: 1. positive, or  2. historical edge
            # NOTE: the selected edges are certain RANDOM edges (no historical edges are randomly selected)
            if ((selected_rnd_src, selected_rnd_dst) not in current_split_e_dict) and \
                    ((selected_rnd_src, selected_rnd_dst) not in history_e_dict):
                neg_rnd_source.append(selected_rnd_src)
                neg_rnd_dest.append(selected_rnd_dst)
                num_selected_smp_rnd += 1

        return neg_rnd_source, neg_rnd_dest

    def get_pos_subcat_indices(self, current_split_start_ts, pos_source_l, pos_dest_l):
        """
        return the indices of the inductive & historical positive edges
        """
        history_e_dict = self.get_edges_in_time_interval(self.ts_init, current_split_start_ts)
        pos_induc_idx_l, pos_hist_idx_l = [], []
        for idx, (src, dst) in enumerate(zip(pos_source_l, pos_dest_l)):
            if (src, dst) in history_e_dict:
                pos_hist_idx_l.append(idx)
            else:
                pos_induc_idx_l.append(idx)

        return pos_induc_idx_l, pos_hist_idx_l

    def reset_random_state(self, new_seed=None):
        if new_seed is not None:
            self.seed = new_seed

        self.random_state = np.random.RandomState(self.seed)
