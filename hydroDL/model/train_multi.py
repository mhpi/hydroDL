import numpy as np
import torch
import time
import os
import hydroDL
import pandas as pd
import torch_scatter
import json
from hydroDL.post.stat import statError
from hydroDL.post.plot import plotBoxFigMulti
from hydroDL.utils.norm import trans_norm


def ext_multi_index(path_json):
    index_json = json.load(open(os.path.join(path_json, "index_bsg_fsg.json"), "r"))

    num_json = [len(x) for x in index_json]
    num_json_acum = np.cumsum(num_json)

    str_index = num_json_acum[0:-1]
    str_index = np.insert(str_index, 0, 0)
    end_index = num_json_acum

    index_flat = [x for y in index_json for x in y]
    multi_index = index_json
    multi_index_flat = index_flat
    multi_str_index = str_index
    multi_end_index = end_index
    multi_index_num = num_json

    return multi_index, multi_index_flat, multi_str_index, multi_end_index, multi_index_num


def trainModel(model,
               # x,
               # y,
               # c,
               satellite_data,
               site_data,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               mode='seq2seq',
               bufftime=0,
               opt=None,
               ):
    path_json = opt["root_DB"]
    saveFolder = opt["out"]
    if os.path.exists(saveFolder):
        pass

    _, _, multi_str_index, multi_end_index, multi_index_num = ext_multi_index(path_json)

    batchSize, rho = miniBatch

    (x_fine, y_coarse, c_fine) = satellite_data
    (x_site, y_site, c_site) = site_data

    ngrid, nt, nx = y_coarse.shape
    ngrid = len(multi_index_num)
    ngauge = len(json.load(open(os.path.join(path_json, "index_bsg_nv2.json"))))

    grid_insitu_ratio = 0.5

    # nIterEpGauge = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngauge / nt)))
    # nIterEp = nIterEpGauge * (1 + ratio_int)

    if c_fine is not None:
        nx = nx + c_fine.shape[-1]
    if batchSize >= ngrid:
        batchSize = ngrid

    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt - bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')

    # For faster training
    x_fine = speed_up(x_fine,c_fine, x_type="fine", path_json=path_json)
    y_coarse = speed_up(y_coarse,x_type="coarse", path_json=path_json)
    x_site = speed_up(x_site,c_site, x_type="site", path_json=path_json)
    y_site = speed_up(y_site,x_type="site", path_json=path_json)

    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            iGrid_insitu, iT_insitu = randomIndex(ngauge, nt, [int(batchSize * (grid_insitu_ratio/(1-grid_insitu_ratio))), rho])

            # select datasets from satellite
            xTrainFine = selectSubset_multi(x_fine, iGrid, iT, rho, iIter, isX=True, path_json=path_json, x_type="fine")
            yTrainCoarse = selectSubset_multi(y_coarse, iGrid, iT, rho, iIter, path_json=path_json, x_type="coarse")

            # select datasets from in-situ
            xTrainSite = selectSubset_multi(x_site, iGrid_insitu, iT_insitu, rho, iIter, path_json=path_json, x_type="site")
            yTrainSite = selectSubset_multi(y_site, iGrid_insitu, iT_insitu, rho, iIter, path_json=path_json, x_type="site")

            #
            yP = model(xTrainFine.to("cuda"))
            yP_site = model(xTrainSite.to("cuda"))

            # mean method for yhat
            idx_str = np.load('tmp/multi_scale/trn_idx_str_' + str(iIter) + '.npy')
            idx_end = np.load('tmp/multi_scale/trn_idx_end_' + str(iIter) + '.npy')
            yP = yP.permute([1, 0, 2])
            lst = []
            for i, x in enumerate((idx_end - idx_str).tolist()):
                lst.append(np.full(x, i).tolist())
            labels = torch.LongTensor([x for y in lst for x in y]).cuda()
            res = torch_scatter.scatter(src=yP, index=labels, dim=0, reduce="mean")
            yP = res.permute([1, 0, 2])  # rho, bs*rho, feas

            yP_site[yP_site > 5.13722] = 5.13722
            yP_site[yP_site < -2.2676] = -2.2676
            yP[yP > 5.13722] = 5.13722
            yP[yP < -2.2676] = -2.2676

            loss = lossFun(yP, yTrainCoarse.to("cuda"))
            loss_insitu = lossFun(yP_site, yTrainSite.to("cuda"))
            # print("loss satellite: {}, loss insitu {}".format(loss, loss_insitu))

            loss = (1 - grid_insitu_ratio) * loss + grid_insitu_ratio * loss_insitu

            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()

        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        # save model and loss
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
    if saveFolder is not None:
        rf.close()
    return model


def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model

def speed_up(x,c=None, x_type=None, path_json=None):
    multi_fine_index = json.load(open(os.path.join(path_json, "index_bsg_fsg.json")))
    multi_coarse_index = json.load(open(os.path.join(path_json, "index_bsg_nv.json")))
    multi_site_index = json.load(open(os.path.join(path_json, "index_bsg_nv2.json")))
    if not c is None:
        c = c[:, None, :]  # bs, time, fea
        c = np.repeat(c, x.shape[1], axis=1)
        x = np.concatenate([x, c], axis=-1)

    # speed up
    if x_type in ["fine"]:
        x = x[[item for sublist in multi_fine_index for item in sublist]]
    elif x_type in ["coarse"]:
        x = x[multi_coarse_index]
    elif x_type in ["site"]:
        x = x[multi_site_index]

    return x

def testModel(model,
              satellite_data,
              site_data,
              *,
              batchSize=None,
              filePathLst=None,
              doMC=False,
              outModel=None,
              opt=None):
    path_json = opt["root_DB"]
    _, _, multi_str_index, multi_end_index, multi_index_num = ext_multi_index(path_json)
    (x_fine, y_coarse, c_fine) = satellite_data
    (x_site, y_site, c_site) = site_data

    # For faster testing
    x_fine = speed_up(x_fine, c_fine, x_type="fine", path_json=path_json)
    y_coarse = speed_up(y_coarse, x_type="coarse", path_json=path_json)
    x_site = speed_up(x_site, c_site, x_type="site", path_json=path_json)
    y_site = speed_up(y_site, x_type="site", path_json=path_json)

    ngrid, nt, ny = y_coarse.shape
    nx = x_fine.shape[-1]

    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)

    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    fLst = list()

    # forward for each batch
    for ndx, i in enumerate(range(0, len(iS))):
        # print('batch {}'.format(i))

        multi_index_val = np.array([multi_end_index[x] for x in iE - 1])
        multi_index_val_str = np.insert(multi_index_val[:-1], 0, 0)
        multi_index_val_end = multi_index_val
        xVal = x_fine[multi_index_val_str[ndx]:multi_index_val_end[ndx], :, :]  # pix, time, feas

        yVal = y_coarse[iS[ndx]:iE[ndx], :, :]
        xVal, yVal = np.swapaxes(xVal, 1, 0), np.swapaxes(yVal, 1, 0)  # rho, pix, feas

        # if torch.cuda.is_available():
        #     xTest = xTest.cuda()
        xVal = torch.from_numpy(xVal).float().to("cuda")

        yP = model(xVal)

        yP = yP.permute([1, 0, 2])
        # mean method for yP
        idx_end = (multi_end_index - multi_str_index)
        lst = []
        for i, x in enumerate(idx_end[iS[ndx]:iE[ndx]]):
            lst.append(np.full(x, i).tolist())
        labels = torch.LongTensor([x for y in lst for x in y]).to("cuda")
        res = torch_scatter.scatter(src=yP, index=labels, dim=0, reduce="mean")
        yP = res.permute([1, 0, 2])  # rho, bs*rho, feas

        if ndx == 0:
            yOut = yP
        else:
            # print(yOut.shape,yP.shape)
            yOut = torch.cat((yOut, yP), dim=1)  # rho,pix,feas

    xInsitu, yInsitu = np.swapaxes(x_site, 1, 0), np.swapaxes(y_site, 1, 0)
    xInsitu = torch.from_numpy(xInsitu).float().to("cuda")
    yP_insitu = model(xInsitu)

    # save output

    model.zero_grad()
    torch.cuda.empty_cache()

    yOut = np.squeeze(yOut.detach().cpu().numpy()).swapaxes(0, 1)
    yOut_insitu = np.squeeze(yP_insitu.detach().cpu().numpy()).swapaxes(0, 1)
    y_coarse = np.squeeze(y_coarse)
    y_site = np.squeeze(y_site)

    # denormalized
    var_s = opt["target"][0]
    rootDB = opt["root_DB"]

    # calculate metrics
    output_fig1 = os.path.join(opt["out"], "metrics.jpg")
    output_fig2 = os.path.join(opt["out"], "metrics_insitu.jpg")
    varLst = ['Bias', 'RMSE', 'ubRMSE', 'Corr']
    median_dict_in_situ = metrics(yOut_insitu, y_site, varLst, var_s, output_fig2)

    return median_dict_in_situ

def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0 + bufftime, nt - rho, [batchSize])
    return iGrid, iT

def selectSubset_multi(x, iGrid, iT, rho, iIter, *, c=None, isX=False, path_json=None, x_type=None):
    _, _, multi_str_index, multi_end_index, multi_index_num = ext_multi_index(path_json)
    if isX:
        nx = x.shape[-1]
        batchSize = iGrid.shape[0]
        idx_a = np.array([multi_index_num[x] for x in iGrid])
        idx_a_cum = np.cumsum(idx_a)
        idx_str = idx_a_cum[0:-1]
        idx_str = np.insert(idx_str, 0, 0)
        idx_end = idx_a_cum
        reFolder('tmp/multi_scale/')
        np.save('tmp/multi_scale/trn_iGrid_' + str(iIter) + '.npy', iGrid)
        np.save('tmp/multi_scale/trn_idx_str_' + str(iIter) + '.npy', idx_str)
        np.save('tmp/multi_scale/trn_idx_end_' + str(iIter) + '.npy', idx_end)

        num_pix = idx_a.sum()
        xTensor = torch.zeros([rho, num_pix, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[multi_str_index[iGrid[k]]:multi_end_index[iGrid[k]], np.arange(iT[k], iT[k] + rho), :]

            xTensor[:, idx_str[k]:idx_end[k], :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
            if not temp.shape[0] == idx_a[k]:
                print('error')

    else:
        nx = x.shape[-1]
        nt = x.shape[1]
        if x.shape[0] == len(iGrid):  # hack
            iGrid = np.arange(0, len(iGrid))  # hack
            if nt <= rho:
                iT.fill(0)

        if iT is not None:
            batchSize = iGrid.shape[0]
            xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
            for ndx_bs, k in enumerate(range(batchSize)):
                temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
                xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        else:
            if len(x.shape) == 2:
                # Used for local calibration kernel
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            else:
                # Used for rho equal to the whole length of time series
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
                rho = xTensor.shape[0]

    out = xTensor

    return out


def reFolder(fn_s):
    if not os.path.exists(fn_s):
        b = os.path.normpath(fn_s).split(os.sep)
        b = [x + '/' for x in b]
        fn_rec = [''.join(b[0:x[0]]) for x in list(enumerate(b, 1))]
        fn_None = [os.mkdir(x) for x in fn_rec if not os.path.exists(x)]


def metrics(yp, yt, keyLst, var_s, output_path="./"):
    statErr = statError(yp, yt)
    statDictLst = [statErr]
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            data = statDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)
    median_dict = plotBoxFigMulti(dataBox, label1=keyLst, sharey=False, figsize=(12, 5), path_fig=output_path)

    return median_dict

