import torch
import torch.nn as nn
from loss import batch_episym

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x) 
    xx = torch.sum(x**2, dim=1, keepdim=True) 
    id=inner - xx.transpose(2, 1)
    pairwise_distance = -xx -id 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx[:, :, :]


def knn2(x, k,gamma1):
    balance = -2 * torch.matmul(x.transpose(2, 1), x)
    idx = (1-gamma1*(torch.sum(x**2, dim=1, keepdim=True)) +balance - torch.sum(x**2, dim=1, keepdim=True).transpose(2, 1)).topk(k=k, dim=-1,largest=False)[1]
    return idx[:, :, :]

def get_graph_feature(gamma1,far,x, k=9,idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1,num_points) 
    if idx is None:
        if far==1:
            idx_out = knn2(x, k,gamma1) 
        else:
            idx_out = knn(x, k=k)  
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base 

    idx = idx.view(-1) 

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() 
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous() 
    return feature

class ResFormer_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResFormer_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.left0 = nn.Sequential(
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.left1 = nn.Sequential(

            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu1 = nn.ReLU()
        self.left2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu2 = nn.ReLU()
        self.left3 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu3 = nn.ReLU()
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        x0 = self.left0(x1)
        out1 = self.left1(x0)
        out1 = self.relu1(out1) + x1
        out2 = self.left2(out1)
        out2 = self.relu2(out2) + out1 + x1
        out3 = self.left3(out2)
        out3 = self.relu3(out3 + x1)
        return out3
def batch_symeig(X):
    
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    
    mask = logits[:, 0, :, 0] 
    weights = logits[:, 1, :, 0] 

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class KNN_Block(nn.Module):
    def __init__(self, far,knn_num=9, in_channel=128,):
        super(KNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.far=far
        self.gamma1 = nn.Parameter(torch.ones(1))
        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), 
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)), 
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), 
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)), 
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        
        B, _, N, _ = features.shape
        g=1/(self.gamma1*self.gamma1)
        out = get_graph_feature(g,self.far,features, k=self.knn_num)
        out = self.conv(out) 
        return out

class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1) 
        A = torch.bmm(w.transpose(1, 2), w) 
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size() 
        with torch.no_grad():
            A = self.attention(w) 
            I = torch.eye(N).unsqueeze(0).to(x.device).detach() 
            A = A + I 
            D_out = torch.sum(A, dim=-1) 
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D) 
            L = torch.bmm(D, A)
            L = torch.bmm(L, D) 
        out = x.squeeze(-1).transpose(1, 2).contiguous() 
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous() 

        return out

    def forward(self, x, w):
        
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        y=self.conv2(out)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

class ContextNorm(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
class OANet(nn.Module):
    def __init__(self, net_channels,clusters):
        nn.Module.__init__(self)
        self.l1_1 = []
        channels = net_channels
        self.layer_num=6
        l2_nums = clusters
        self.l1_1.append(ContextNorm((channels), channels))
        for _ in range(self.layer_num // 2 - 1):
            self.l1_1.append(ContextNorm(channels))
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.ResFormer=ResFormer_Block(channels*2,channels,pre=True)
        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l2 = nn.Sequential(*self.l2)

    def forward(self, x):
        x1_1 = self.l1_1(x)  
        x_down = self.down1(x1_1)  
        x2 = self.l2(x_down)  
        x_up = self.up1(x1_1, x2)  
        out = torch.cat([x1_1, x_up], dim=1)
        out=self.ResFormer(out)
        return out

class CFS(nn.Module):
    def __init__(self, in_channel,out_channels):
        nn.Module.__init__(self)
        self.att1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels, kernel_size=1),
        )
        self.attq1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attk1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.trans_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.att2 = ResFormer_Block(in_channel*2,out_channels,pre=True)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.after_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x_row, x_local):
        
        x_local = self.att1(x_local)  
        q = self.attq1(x_row)  
        k = self.attk1(x_local)  
        v = self.attv1(x_local)  
        att = torch.mul(q, k)  
        att = torch.softmax(att, dim=3)  
        qv = torch.mul(att, v)  
        out_local=torch.cat((x_row,qv),dim=1)
        out_local=self.att2(out_local)
        out = x_row + self.gamma1 * out_local  
        return (out+out.mean(dim=1,keepdim=True))*0.5

class PAM_Module(nn.Module):
    def __init__(self,out_channels):
        nn.Module.__init__(self)
        self.query=nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax=nn.Softmax (dim=-1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

    def forward(self, x,y):
        q=self.query(y)
        k=self.key(y)
        energy=torch.mul(q, k)
        attention=self.softmax(energy)
        v = self.value(x)
        out=self.gamma1 *attention+v
        return out

class CGR_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(CGR_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), 
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN_Block(self.out_channel)
        self.cluster=250
        self.cfs = CFS(self.out_channel, self.out_channel)
        self.pam = PAM_Module(self.out_channel)
        self.RKR = nn.Sequential(
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            KNN_Block(0,self.k_num, self.out_channel),
            OANet(self.out_channel, self.cluster),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.gsc = nn.Sequential(
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            #Construct Consistency Graph
            KNN_Block(1,6, self.out_channel),
            OANet(self.out_channel, self.cluster),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
            ResFormer_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.gamma1 = nn.Parameter(torch.ones(1))

        if self.predict == True:
            self.embed_2 = ResFormer_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] 
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) 
            w_out = torch.gather(weights, dim=-1, index=indices) 
        indices = indices.view(B, 1, -1, 1) 

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
      
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out) 
        out = self.RKR(out)  
        w0 = self.linear_0(out).view(B, -1) 
        out_g = self.gcn(out, w0.detach())  
        out1 = self.cfs(out_g, out) 
        out2 = self.gsc(out1)
        out2=self.pam(out2,out_g)
        w1 = self.linear_1(out2).view(B, -1) 

        if self.predict == False: 
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)] 
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else: 
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out) 
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class CGRNet(nn.Module):
    def __init__(self, config):
        super(CGRNet, self).__init__()

        self.cgr_0 = CGR_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)
        self.cgr_1 = CGR_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.cgr_0(x, y) 

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1) 
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1) 
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1) 

        x2, y2, ws1, w_ds1, e_hat = self.cgr_1(x_, y1) 

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat) 
        #print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

