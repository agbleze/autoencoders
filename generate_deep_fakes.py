
#%%   ##### generating deep fake  #####
import os
if not os.path.exists('Faceswap-Deepfake-Pytorch'):
    !wget -q https://www.dropbox.com/s/5ji7jl7httso9ny/person_images.zip
    !wget -q https://raw.githubusercontent.com/sizhky/deep-fake-util/main/random_warp.py
    !unzip -q person_images.zip

#%%
!mkdir cropped_faces_personA
!mkdir cropped_faces_personB

#%%
from torch_snippets import *
from random_warp import get_training_data


#%%
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#%%
def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if (len(faces)>0):
        for (x,y,w,h) in faces:
            img2 = img[y:(y+h), x:(x+h), :]
        img2 = cv2.resize(img2, (256,256))
        return img2, True
    else:
        return img, False

#%% ## crop images of A and B into sperate folder
def crop_images(folder):
    images = Glob(folder + "/*.jpg")
    for i in range(len(images)):
        img = read(images[i], 1)
        img2, face_detected = crop_face(img)
        if (face_detected==False):
            continue
        else:
            cv2.imwrite("cropped_faces_" + folder + "/" + str(i) + ".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
crop_images("personA")
crop_images("personB")
        
#%%
class ImageDataset(Dataset):
    def __init__(self, items_A, items_B) -> None:
        self.items_A = np.concatenate([read(f,1)[None] for f in items_A])/255.
        self.items_B = np.concatenate([read(f, 1)[None] for f in items_B])/255.
        self.items_A += self.items_B.mean(axis=(0,1,2)) - self.items_A.mean(axis=(0, 1, 2))
        
    def __len__(self):
        return min(len(self.items_A), len(self.items_B))
    
    def __getitem__(self, ix):
        a, b = choose(self.items_A), choose(self.items_B)
        return a, b
    
    def collate_fn(self, batch):
        imsA, imsB = list(zip(*batch))
        imsA, targetA = get_training_data(imsA, len(imsA))
        imsB, targetB = get_training_data(imsB, len(imsB))
        imsA, imsB, targetA, targetB = [torch.Tensor(i).permute(0,3,1,2).to(device)
                                        for i in [imsA, imsB, targetA, targetB]]
        return imsA, imsB, targetA, targetB

# %%
a = ImageDataset(Glob("cropped_faces_personA"), Glob("cropped_faces_personB"))
x = DataLoader(a, batch_size=32, collate_fn=a.collate_fn)


#%%
inspect(*next(iter(x)))
for i in next(iter(x)):
    subplots(i[:8], nc=4, sz=(4,2))


# %%
