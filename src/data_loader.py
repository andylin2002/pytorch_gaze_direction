import os
from PIL import Image
import torchvision.transforms as transforms

class ImageData(object):

    def __init__(self, load_size, channels, data_path, ids):

        self.load_size = load_size
        self.channels = channels
        self.ids = ids

        self.data_path = data_path
        file_names = [f for f in os.listdir(data_path)
                       if f.endswith('.jpg')]

        self.file_dict = {}
        for f_name in file_names:
        # Extract attributes: identity, head_pose, side
            fields = f_name.split('.')[0].split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict:
                self.file_dict[key] = []
            self.file_dict[key].append(f_name) # f_name is the full name of the photo

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g,
        augment
    ):
        
        def _to_image(file_name, augment):
            # Load image
            img = Image.open(file_name).convert("RGB" if self.channels == 3 else "L")
            
            # Define basic transformation pipeline
            transform = transforms.Compose([
                transforms.Resize((self.load_size, self.load_size)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
            ])

            # Define data augmentation pipeline
            augmentation = transforms.Compose([
                transforms.RandomErasing(p=0.3, scale=(0.05, 0.2))
            ])
            
            # Apply transformations
            img = transform(img)

            # Apply augmentation (training data only)
            if augment:
                img = augmentation(img)
            
            return img

        
        image = _to_image(filename, augment)
        image_t = _to_image(filename_t, augment=False)

        return image, angles_r, labels, image_t, angles_g
    
    def preprocess(self):

        for key, file_list in self.file_dict.items(): # Photos of the same person from different angles
            if len(file_list) == 1: # Skip if there is only one image
                continue
            
            idx = int(key.split('_')[0])
            flip = 1
            if key.split('_')[-1] == 'R': # Check if it's the right eye
                flip = -1
            
            for f_r in file_list: # For each photo of a key (e.g. 0010_0P_R)
                file_path = os.path.join(self.data_path, f_r)
                h_angle_r = flip * float(f_r.split('_')[-2].split('H')[0]) / 15.0 # Horizontal angle
                v_angle_r = float(f_r.split('_')[-3].split('V')[0]) / 10.0 # Vertical angle

                for f_g in file_list:
                    file_path_t = os.path.join(self.data_path, f_g)
                    h_angle_g = flip * float(f_g.split('_')[-2].split('H')[0]) / 15.0
                    v_angle_g = float(f_g.split('_')[-3].split('V')[0]) / 10.0
                    
                    if idx <= self.ids: # Training set
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                    else: # Test set
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx - 1)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])

        print('\nFinished preprocessing the dataset...')