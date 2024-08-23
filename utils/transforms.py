import torchio as tio

def preprocess(num_classes):
    return tio.Compose([
                tio.OneHot(num_classes=num_classes),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.ToCanonical(), # Reorder the data to be closest to canonical (RAS+) orientation.
                tio.Resample('t1'), # Make sure all seg have same affine as t1
                tio.Resample(1),
                tio.EnsureShapeMultiple((32,32,32), method='crop'),# for the U-Net : doit être un multiple de 2**nombre de couches
                ])
        
def augment():
    return tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=(0.5,1),degrees=20, translation=(-10,10)): 0.8,
                    tio.RandomElasticDeformation(num_control_points = 12, max_displacement = 4): 0.2,
                    },
                    p=0.75,
                ),
                tio.RandomMotion(p=0.2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),  # Change contrast 
                tio.RandomFlip(axes=('LR',), flip_probability=0.2),
                tio.RandomBiasField(coefficients = 0.5, order = 3, p=0.5),
                tio.RandomNoise(mean = 0, std=(0.005, 0.1), p=0.25),
                                ])