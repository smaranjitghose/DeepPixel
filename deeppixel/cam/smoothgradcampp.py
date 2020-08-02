import torch
import torch.nn.functional as F

class SmoothGradCAMpp(GradCAM):

    """
        Smooth Grad-CAM++, inherit from GradCAM
    """
  
  def __init__(self, model_dict):
    super(SmoothGradCAMpp, self).__init__(model_dict)

  def forward(self, input_image, class_idx=None, param_n=35, mean=0, sigma=2, retain_graph=False):

    b, c, h, w = input_image.size()

    logit = self.model_arch(input_image)
    if class_idx is None:
        score = logit[:, logit.max(1)[-1]].squeeze()
    else:
        score = logit[:, class_idx].squeeze()

    
    if torch.cuda.is_available():
      score = score.cuda()
      logit = logit.cuda()

    self.model_arch.zero_grad()
    score.backward(retain_graph=retain_graph)
    gradients = self.gradients['value']
    activations = self.activations['value']
    b, k, u, v = gradients.size()

    if torch.cuda.is_available():
      activations = activations.cuda()
      gradients = gradients.cuda() 

    #HYPERPARAMETERS (can be changed, have a look at the arguments!)
    #mean = 0
    #param_n = 35
    #param_sigma_multiplier = 2

    grad_2, grad_3 = torch.zeros_like(activations), torch.zeros_like(activations)

    for i in range(param_n):
      
      noise = Variable(input_image.data.new(input_image.size()).normal_(0,param_sigma_multiplier**2))

      noisy_input = input_image + noise
      
      if torch.cuda.is_available():
        noisy_input = noisy_input.cuda()

      out = self.model_arch(noisy_input)
      score = out[:, out.max(1)[-1]].squeeze()
      self.model_arch.zero_grad()
      score.backward(retain_graph=retain_graph)
      gradient = self.gradients['value']
      
      grad_2.add_(gradient.pow(2))
      grad_3.add_(gradient.pow(3))

    grad_2.div_(param_n)
    grad_3.div_(param_n)

    # Alpha coefficient for each pixel

    global_sum = activations.view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_num = grad_2
    alpha_denom = grad_2.mul(2) + global_sum.mul(grad_3)

    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom + 1e-7)
    positive_gradients = F.relu(score.exp() * gradients)
    weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
      return self.forward(input, class_idx, retain_graph)

