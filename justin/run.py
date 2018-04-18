import justin
import torch

use_cuda = torch.cuda.is_available()
run = justin.justin(style_img_dir='images/style/sub.JPG', content_img_dir='images')
run.train_network()
## output = run.feed_forward('1.jpeg')
## run.imshow(output.data)
