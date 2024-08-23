import torchio as tio
import pandas as pd
from typing import Any

def get_subjects(image_paths, label_paths=None, subsample=False): 
    subjects = []

    if label_paths == None :
        for image_path in image_paths:
            subject = MySubject(
                t1=tio.ScalarImage(image_path),
                subject=str(image_path).split("/")[-1].split(".nii.gz")[0]
            )
            subjects.append(subject)
    else:
        for (image_path, label_path) in zip(image_paths, label_paths):
            subject_id = str(image_path).split("/")[-1].split("_")[0]
            if subsample == True :
                df = pd.read_csv("/home/emma/Projets/stroke_lesion_segmentation_v2/config_files/df_atlas.csv")
                sub_df = df[df['Subject ID'] == subject_id]
                if df.iloc[0, sub_df.columns.get_loc('lesion_size')] > 0.57 and sub_df.iloc[0, df.columns.get_loc('mean_lesion_intensity')] < 0.64:
                    subject = MySubject(
                        t1=tio.ScalarImage(image_path),
                        seg=tio.LabelMap(label_path),
                        subject=subject_id
                    )
                    subjects.append(subject)
            else:
                subject = MySubject(
                        t1=tio.ScalarImage(image_path),
                        seg=tio.LabelMap(label_path),
                        subject=subject_id
                    )
                subjects.append(subject)
    
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
    
    