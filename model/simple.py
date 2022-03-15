import torch
import torch.nn as nn

class EncoderSimple(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, dim_feature=9, dim_code=8, threshold_ReLU=0.2):
        super(EncoderSimple, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 1),
			nn.LeakyReLU(threshold_ReLU, inplace=True),	
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(dim_feature * 1, dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 2),
			nn.LeakyReLU(threshold_ReLU, inplace=True),	
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(dim_feature * 2, dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 4),
			nn.LeakyReLU(threshold_ReLU, inplace=True),		
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(dim_feature * 4, dim_feature * 8, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 8),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(dim_feature * 8, dim_feature * 16, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 16),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(dim_feature * 16, dim_feature * 32, kernel_size=3, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 32),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        
        layers  = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        self.encoder    = nn.ModuleList(layers)

        self.initialize_weight()

    def initialize_weight(self):
        for m in self.encoder.modules():
			
            if isinstance(m, nn.Conv2d):
				
				#nn.init.uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)
				
                if m.bias is not None:
	
                    nn.init.constant_(m.bias, 0.1)
					
            elif isinstance(m, nn.BatchNorm2d):
				
                nn.init.constant_(m.weight, 1)

                if m.bias is not None:
    	
                    nn.init.constant_(m.bias, 0.1)

            elif isinstance(m, nn.Linear):
				
                #nn.init.uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
					
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        skip = []
        for layer in self.encoder:
            x   = layer(x)
            skip.append(x)

        skip.reverse()

        return x, skip

        
class DecoderSimple(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, dim_feature=9, dim_code=8, threshold_ReLU=0.2):
        super(DecoderSimple, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 32, dim_feature * 16, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 16),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 16, dim_feature * 8, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 8),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 8, dim_feature * 4, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 4),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 4, dim_feature * 2, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 2),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 2, dim_feature * 1, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(dim_feature * 1),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
        )

        self.layer6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.Conv2d(dim_feature * 1, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_channel),
			nn.Sigmoid(),
        )

        
        layers  = [self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]
        self.decoder    = nn.ModuleList(layers)

        self.initialize_weight()

    def initialize_weight(self):
        for m in self.decoder.modules():
            
            if isinstance(m, nn.Conv2d):
                
                #nn.init.uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                
                if m.bias is not None:
    
                    nn.init.constant_(m.bias, 0.1)
                    
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant_(m.weight, 1)

                if m.bias is not None:
        
                    nn.init.constant_(m.bias, 0.1)

            elif isinstance(m, nn.Linear):
                
                #nn.init.uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, skip):

        x   = self.decoder[0](x)
        for pair in zip(self.decoder[1:], skip[1:]):
            layer   = pair[0]
            # x       = layer(x + pair[1])
            x       = layer(x)

        return x