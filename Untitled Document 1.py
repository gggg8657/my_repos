from ..models.OpenGraphAU.model.MEFL import MEFARG
from ..models.OpenGraphAU.utils import load_state_dict as AU_load_state_dict
from ..models.OpenGraphAU.utils import *
from ..models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
import os
from torchvision.transforms.functional import to_pil_image
class AU_Feature_Loss(nn.Module):
    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone=get_config().arc):
        super(AU_Feature_Loss, self).__init__()
        self.auconfig = get_config()
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)
        
        self.aufeat = MEFARG(num_main_classes, num_sub_classes, backbone).to('cuda')
        self.aufeat = AU_load_state_dict(self.aufeat, self.auconfig.resume).to('cuda')
        self.aufeat.eval()
        self.criterion = nn.MSELoss()

    '''
    for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
    '''
    def map_threshold(self, value: float):
        # if value <= 0.2:
        #     return 2
        # elif value <= 0.4:
        #     return 4
        # elif value <= 0.6:
        #     return 6
        # elif value <= 0.8:
        #     return 8
        # else:
        #     return 10
        threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(threshold)):
            if value < threshold[i]:
                return threshold[i]*10
        
    def forward(self, input_image, target_image):
        img_transform = image_eval()
        loss_ = 0
        src_feature_l = []
        tar_feature_l = []
        src_chin_weighted_l = []
        tar_chin_weighted_l = []
        src_dimp_weighted_l = []
        tar_dimp_weighted_l = []

        BatchSize = target_image.size(0)
        for i in range(8):    
            input_src = input_image[i]
            target_src = target_image[i]
            # input image num is 16 , target image num is 16 
            input_image_ = img_transform(to_pil_image(input_src)).unsqueeze(0)
            input_features = self.aufeat(input_image_.cuda())
            target_image_ = img_transform(to_pil_image(target_src)).unsqueeze(0)        
            target_features = self.aufeat(target_image_.cuda())
            
            src_chin_score = input_features[1][0][14] #AU 17 chin raiser
            # src_chin_score = src_chin_score * self.map_threshold(src_chin_score)
            tar_chin_score = target_features[1][0][14] 
            # tar_chin_score = tar_chin_score * self.map_threshold(tar_chin_score)
            src_dimp_score = input_features[1][0][11] #AU 14 dimpler
            # src_dimp_score = src_dimp_score * self.map_threshold(src_dimp_score)
            tar_dimp_score = target_features[1][0][11]
            # tar_dimp_score = tar_dimp_score * self.map_threshold(tar_dimp_score)
            
            src_chin_weighted_l.append(src_chin_score)
            tar_chin_weighted_l.append(tar_chin_score)
            src_dimp_weighted_l.append(src_dimp_score)
            tar_dimp_weighted_l.append(tar_dimp_score)
            
            src_feature_l.append(input_features[1])
            tar_feature_l.append(target_features[1])
        for i in range(8):
            chin_loss = self.criterion(src_chin_weighted_l[i], tar_chin_weighted_l[i])
            dimp_loss = self.criterion(src_dimp_weighted_l[i], tar_dimp_weighted_l[i])
            loss = self.criterion(src_feature_l[i], tar_feature_l[i])
            loss_ = loss_+loss 
        return loss_, chin_loss*2.0, dimp_loss*10.0

