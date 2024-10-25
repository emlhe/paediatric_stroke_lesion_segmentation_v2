import torchio as tio
import pandas as pd
from typing import Any

def get_subjects(image_paths, label_paths=None, subsample=False, brain_mask_paths=None): 
    subjects = []
    i=0
    for image_path in image_paths:
        subject_id = str(image_path).split("/")[-1].split("_")[0]
        if subsample == True :
            df = pd.read_csv("/home/emma/Projets/stroke_lesion_segmentation_v2/config_files/df_atlas.csv")
            df_other_lesions = pd.read_csv("/home/emma/Projets/stroke_lesion_segmentation_v2/config_files/other_lesions_atlas.csv")
            if not subject_id in df_other_lesions['subject'].to_list():
                sub_df = df[df['Subject ID'] == subject_id]
                if not (sub_df.iloc[0, sub_df.columns.get_loc('lesion_size')] < 0.57 and sub_df.iloc[0, sub_df.columns.get_loc('mean_lesion_intensity')] > 0.64):
                    subject = MySubject(
                        t1=tio.ScalarImage(image_path),
                        subject=subject_id, 
                    )
        else:
            subject = MySubject(
                    t1=tio.ScalarImage(image_path),
                    subject=subject_id
                )
        if brain_mask_paths != None:
            subject.add_image(tio.LabelMap(brain_mask_paths[i]), "brain_mask")
        if label_paths != None:
            subject.add_image(tio.LabelMap(label_paths[i]), "seg")
        subjects.append(subject)
        i+=1
    return subjects


class MySubject(tio.Subject):
    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:

        out_dict = dict()

        for key, value in self.items():
            # if value is an image, extract the tensor
            if isinstance(value, tio.Image):
                value = {'data':value.data,'affine':value.affine} 

            out_dict[key] = value

        return out_dict
    
    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-5
        kwargs['absolute_tolerance'] = 1e-5
        return super().check_consistent_attribute(*args, **kwargs)
    
    