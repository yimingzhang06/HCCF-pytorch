import torch
import torch.nn as nn
from Params import args

torch.manual_seed(666)

def LeakyRelu(data):
  #global leaky
  ret = torch.maximum(args.leaky*data, data)
  return ret

class FC(nn.Module):
  def __init__(self, inputDim, outDim, Bias = False, actFunc = None):
    super(FC,self).__init__()
    initializer = nn.init.xavier_normal_
    self.W_fc = nn.Parameter(initializer(torch.empty(inputDim, outDim).cuda()))

  def forward(self, inp, droprate = 0):
    #W = self.W_fc.weight
    fc1 = inp @ self.W_fc
    ret = fc1
    ret = LeakyRelu(ret)
    return ret

class hyperPropagate(nn.Module):
  def __init__(self,inputdim):
    super(hyperPropagate, self).__init__()
    self.inputdim = inputdim
    self.fc1 = FC(self.inputdim,args.hyperNum,actFunc = 'leakyRelu').cuda()
    self.fc2 = FC(self.inputdim,args.hyperNum,actFunc = 'leakyRelu').cuda()
    self.fc3 = FC(self.inputdim,args.hyperNum,actFunc = 'leakyRelu').cuda()
    #self.actFunc = nn.LeakyReLU(negative_slope=args.leaky) 

  def forward(self,lats,adj):
    lat1 = LeakyRelu(torch.transpose(adj,0,1) @ lats) #shape adj:user,hyperNum lats:user,latdim lat1:hypernum,latdim
    lat2 = torch.transpose(self.fc1(torch.transpose(lat1,0,1)),0,1) + lat1 #shape hypernum,latdim
    lat3 = torch.transpose(self.fc2(torch.transpose(lat2,0,1)),0,1) + lat2
    lat4 = torch.transpose(self.fc3(torch.transpose(lat3,0,1)),0,1) + lat3
    ret = adj @ lat4
    ret = LeakyRelu(ret)
    return ret

class weight_trans(nn.Module):
  def __init__(self):
    super(weight_trans, self).__init__()
    initializer = nn.init.xavier_normal_
    self.W = nn.Parameter(initializer(torch.empty(args.latdim, args.latdim).cuda()))

  def forward(self,normalize):
    ret = normalize @ self.W
    return ret

class HCCF(nn.Module):
  def __init__(self, adj_py, tpAdj_py):
    super(HCCF, self).__init__()
    initializer = nn.init.xavier_normal_
    self.uEmbed0 = nn.Parameter(initializer(torch.empty(args.user, args.latdim).cuda()))
    self.iEmbed0 = nn.Parameter(initializer(torch.empty(args.item, args.latdim).cuda()))
    self.uhyper = nn.Parameter(initializer(torch.empty(args.latdim, args.hyperNum).cuda()))
    self.ihyper = nn.Parameter(initializer(torch.empty(args.latdim, args.hyperNum).cuda()))

    self.adj = adj_py.cuda()#shape user,item
    self.tpadj = tpAdj_py.cuda()#shape item,user

    self.hyperULat_layers = nn.ModuleList()
    self.hyperILat_layers = nn.ModuleList()
    self.weight_layers = nn.ModuleList()
    
    for i in range(args.gnn_layer):
      self.hyperULat_layers.append(hyperPropagate(args.hyperNum)) #shape hyperNum,hyperNum
      self.hyperILat_layers.append(hyperPropagate(args.hyperNum)) #shape hyperNum,hyperNum
      self.weight_layers.append(weight_trans())


  def messagePropagate(self, lats, adj):
    return LeakyRelu(torch.sparse.mm(adj, lats))

  def calcSSL(self, hyperLat, gnnLat):
    posScore = torch.exp(torch.sum(hyperLat * gnnLat, dim = 1) / args.temp)
    negScore = torch.sum(torch.exp(gnnLat @ torch.transpose(hyperLat, 0, 1) / args.temp), dim = 1)
    uLoss = torch.sum(-torch.log(posScore / (negScore + 1e-8) + 1e-8))
    return uLoss

  def Regularize(self, reg, method = 'L2'):
    ret = 0.0
    for i in range(len(reg)):
        ret += torch.sum(torch.square(reg[i]))
    return ret

  def edgeDropout(self, mat, drop):
    def dropOneMat(mat):
      indices = mat._indices().cpu()
      values = mat._values().cpu()
      shape = mat.shape
      newVals = nn.functional.dropout(values, p = drop)
      return torch.sparse.FloatTensor(indices, newVals, shape).to(torch.float32).cuda()
    return dropOneMat(mat)

  def forward_test(self):
    uEmbed0 = self.uEmbed0
    iEmbed0 = self.iEmbed0
    uhyper = self.uhyper
    ihyper = self.ihyper

    uuHyper = uEmbed0 @ uhyper#shape user,hyperNum
    iiHyper = iEmbed0 @ ihyper#shape item,hyperNum

    ulats = [uEmbed0]
    ilats = [iEmbed0]

    for i in range(args.gnn_layer):
      ulat = self.messagePropagate(ilats[-1], self.edgeDropout(self.adj, drop = 0))
      ilat = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpadj, drop = 0))
      hyperULat = self.hyperULat_layers[i](ulats[-1],nn.functional.dropout(uuHyper, p = 0))
      hyperILat = self.hyperILat_layers[i](ilats[-1],nn.functional.dropout(iiHyper, p = 0))

      ulats.append(ulat + hyperULat + ulats[-1])
      ilats.append(ilat + hyperILat + ilats[-1])

    ulat = sum(ulats)
    ilat = sum(ilats)
    return ulat, ilat

  def forward(self, uids, iids, droprate = args.droprate):
    uEmbed0 = self.uEmbed0
    iEmbed0 = self.iEmbed0
    uhyper = self.uhyper
    ihyper = self.ihyper
    gnnULats = []
    gnnILats = []
    hyperULats = []
    hyperILats = []

    ulats = [uEmbed0]
    ilats = [iEmbed0]
    for i in range(args.gnn_layer):
      ulat = self.messagePropagate(ilats[-1], self.edgeDropout(self.adj, drop = droprate))
      ilat = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpadj, drop = droprate))
      hyperULat = self.hyperULat_layers[i](ulats[-1],nn.functional.dropout(uEmbed0 @ uhyper, p = droprate))# / (1 - droprate))
      hyperILat = self.hyperILat_layers[i](ilats[-1],nn.functional.dropout(iEmbed0 @ ihyper, p = droprate))#/ (1 - droprate) )

      gnnULats.append(ulat)
      gnnILats.append(ilat)
      hyperULats.append(hyperULat)
      hyperILats.append(hyperILat)

      ulats.append(ulat + hyperULat + ulats[-1])
      ilats.append(ilat + hyperILat + ilats[-1])

    ulat = sum(ulats)
    ilat = sum(ilats)
    pckUlat = torch.index_select(ulat, 0, uids)
    pckIlat = torch.index_select(ilat, 0, iids)
    preds = torch.sum(pckUlat * pckIlat, dim=-1)

    sslloss = 0
    uniqUids = torch.unique(uids)
    uniqIids = torch.unique(iids)

    for i in range(len(hyperULats)):
      pckHyperULat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(hyperULats[i], 0, uniqUids), p=2, dim=1))# @ self.weight_layers[i].weight
      pckGnnULat = torch.nn.functional.normalize(torch.index_select(gnnULats[i], 0, uniqUids), p=2, dim=1)
      pckhyperILat = self.weight_layers[i](torch.nn.functional.normalize(torch.index_select(hyperILats[i], 0, uniqIids), p=2, dim=1))# @ self.weight_layers[i].weight
      pckGnnILat = torch.nn.functional.normalize(torch.index_select(gnnILats[i], 0, uniqIids), p=2, dim=1)
      uLoss = self.calcSSL(pckHyperULat, pckGnnULat)
      iLoss = self.calcSSL(pckhyperILat, pckGnnILat)
      sslloss += uLoss + iLoss

    return preds, sslloss, self.Regularize([uEmbed0,iEmbed0,uhyper,ihyper])