

class SampleResult:

    def __init__(self, bbox_targets, bbox_targets_weights, bbox_labels, bbox_labels_weights, pos_inds, neg_inds):
        self.bbox_targets = bbox_targets
        self.bbox_targets_weights = bbox_targets_weights
        self.bbox_labels = bbox_labels
        self.bbox_labels_weights = bbox_labels_weights
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_pos_inds(self):
        """int: the number of predictions in this assignment"""
        return len(self.pos_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

