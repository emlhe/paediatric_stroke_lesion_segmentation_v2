from typing import Any

import torch
import torch.utils.data
import torchio as tio

# Code from https://github.com/fepegar/torchio/issues/1179

class SubjectDataLoader(torch.utils.data.DataLoader):

    def __init__(
            self,
            dataset: tio.data.SubjectsDataset,
            **kwargs
    ):

        super().__init__(
            dataset=dataset,
            collate_fn=self._collate,
            **kwargs
        )

    @staticmethod
    def _collate(batch_inputs: list[tio.Subject]) -> dict[str, Any]:

        # empty dict to store batch output
        batch_dict = dict()

        # iterate over the keys in the first subject
        for key, first_value in batch_inputs[0].items():

            # get list of the attribute values of each subject in the batch
            batch_attr_value = [batch_input[key] for batch_input in batch_inputs]

            # for image attrs we extract the tensors and stack to create a batched tensor
            if isinstance(first_value, tio.Image):
                batch_attr_value = [attr.data for attr in batch_attr_value]
                batch_attr_value = torch.stack(batch_attr_value, dim=0)

            # for tensor attrs we stack the tensors to create a batched tensor
            elif isinstance(first_value, torch.Tensor):
                batch_attr_value = torch.stack(batch_attr_value, dim=0)

            # update batch output
            batch_dict[key] = batch_attr_value

        return batch_dict


