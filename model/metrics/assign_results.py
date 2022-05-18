

class AssignResult:

    def __init__(self, num_gts, num_bboxes, assigned_gt_inds, max_overlaps, assigned_labels, bboxes, targets):
        self.num_gts = num_gts
        self.num_bboxes = num_bboxes
        self.assigned_gt_inds = assigned_gt_inds
        self.max_overlaps = max_overlaps
        self.assigned_labels = assigned_labels
        self.bboxes = bboxes
        self.targets = targets
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

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

