import torch
import numpy as np

from Model import HCCF
from DataHandler import DataHandler, negSamp, transToLsts, transpose
from Params import args
from TimeLogger import log

torch.manual_seed(666)
np.random.seed(666)

class hccf():
    def __init__(self,handler):
        self.handler = handler
        self.handler.LoadData()

        adj = handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj_py = torch.sparse.FloatTensor(idx, data, shape).to(torch.float32).cuda()
        idx, data, shape = transToLsts(transpose(adj), norm=True)
        self.tpAdj_py = torch.sparse.FloatTensor(idx, data, shape).to(torch.float32).cuda()

        self.curepoch = 0
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
          self.metrics['Train' + met] = list()
          self.metrics['Test' + met] = list()

    def preparemodel(self):
        self.model = HCCF(self.adj_py, self.tpAdj_py).cuda()
        self.opt = torch.optim.Adam(params = self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = self.opt, gamma=args.decayRate)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
        uLocsa = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocsa = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        
        return torch.Tensor(uLocsa).cuda(), torch.Tensor(iLocsa).cuda()

    def trainEpoch(self):
        args.actFunc = 'leakyRelu'

        num = args.user
        #randomly select args.trnNum users(10,000), from args.user(29,601 amazon), as input.
        sfIds = np.random.permutation(args.user)[:args.trnNum]
        epochLoss, epochPreLoss, epochsslloss, epochregloss = [0] * 4
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        self.model.train()

        for i in range(steps):
            st = i * args.batch
            ed = min((i+1) * args.batch, num)
            batIds = sfIds[st: ed]

            uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)
            
            preds, sslloss, regularize = self.model(uLocs.int(),iLocs.int())
            sampNum = uLocs.shape[0] // 2
            posPred = preds[:sampNum]
            negPred = preds[sampNum:sampNum * 2]
            preLoss = torch.sum(torch.maximum(torch.Tensor([0.0]).to(args.device), 1.0 - (posPred - negPred))) / args.batch
            sslloss = args.ssl_reg * sslloss
            regLoss = args.reg * regularize

            loss = preLoss + regLoss + sslloss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % args.decay_step == 0:
                self.scheduler.step()

            epochLoss += loss
            epochPreLoss += preLoss
            epochregloss += args.reg * regularize
            epochsslloss += args.ssl_reg * sslloss
            #log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=False)

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        ret['sslLoss'] = epochsslloss / steps
        ret['regLoss'] = epochregloss / steps

        return ret

    def testEpoch(self):
      self.model.eval()
      with torch.no_grad():
        epochRecall, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        tstNum = 0
        ulat, ilat = self.model.forward_test()
        for i in range(steps):
            st = i * tstBat
            ed = min((i+1) * tstBat, num)
            batIds = ids[st: ed]
            trnPosMask = self.handler.trnMat[batIds].toarray()
            toplocs = self.tstPred(batIds, trnPosMask, ulat, ilat)
            recall, ndcg = self.calcRes(toplocs, self.handler.tstLocs, batIds)
            epochRecall += recall
            epochNdcg += ndcg
            #log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=False)
        ret = dict()
        ret['Recall'] = epochRecall / num
        ret['NDCG'] = epochNdcg / num
      return ret

    def tstPred(self, batIds, trnPosMask, ulat, ilat):
      pckUlat = torch.index_select(ulat, 0, torch.Tensor(batIds).int().to(args.device))
      allPreds = pckUlat @ torch.transpose(ilat, 0, 1)
      allPreds = allPreds.cpu().detach().numpy() * (1 - trnPosMask) - trnPosMask * 1e8
      vals, locs = torch.topk(torch.tensor(allPreds), args.shoot)
      return locs

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig =0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.shoot))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def loadModel(self, loadPath):
        loadPath = loadPath
        checkpoint = torch.load(loadPath)
        self.model = checkpoint['model']
        self.curepoch = checkpoint['epoch']+1
        self.metrics = checkpoint['metrics']

    def saveHistory(self):

        savePath = r'./Model/' + args.data  + r'.pth'
        params = {
            'epoch' : self.curepoch,
            'model' : self.model,
            'metrics' : self.metrics,
        }
        torch.save(params, savePath)


    def run(self):
        self.preparemodel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel(args.load_model)
            stloc = self.curepoch
        else:
            stloc = 0

        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            #print(self.model.hyperULat_layers[0].fc1.W_fc.weight)
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, test))
            if ep % args.tstEpoch == 0:
                self.saveHistory()
            print()
            self.curepoch = ep
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def makePrint(self, name, ep, reses, save):
      ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
      for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
      ret = ret[:-2] + '  '
      return ret

if __name__ == '__main__':
    handler = DataHandler()
    handler.LoadData()
    model=hccf(handler)
    model.run()