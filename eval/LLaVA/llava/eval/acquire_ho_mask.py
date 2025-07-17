import numpy as np
from  PIL import Image
import os
image_name = 'HICO_train2015_00036684.jpg_0' 
h_mask = Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','human_mask.jpg'))
o_mask = Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','object_mask.jpg'))
union_mask = np.maximum(np.array(h_mask)[:,:,0],np.array(o_mask)[:,:,0])
union_mask_3 = np.stack([union_mask,union_mask,union_mask],-1)
Image.fromarray(union_mask_3).save('./retrieved_4.jpg')