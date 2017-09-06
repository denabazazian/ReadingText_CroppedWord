#!/usr/bin/env python
#import sklearn.naive_bayes#


from multiprocessing import Pool
import os.path

import numpy as np #
from commands import getoutput as go
import sys
import time
import numpy.matlib #
from collections import defaultdict
import cPickle
import scipy.stats#

import cv2
import matplotlib.pyplot as plt#


def myread(fname):
    txt=open(fname).read().strip()
    if txt[:3]=='\xef\xbb\xbf':
        txt=txt[3:]
    return txt

def download(url,fname):
    cmd='wget -c '+url+' -O '+fname
    go(cmd)


def get2PointIU(gtMat,resMat):
    gtMat=gtMat.copy()
    resMat=resMat.copy()
    maxProposalsIoU=int(switches['maxProposalsIoU'])
    if maxProposalsIoU>0:
        resMat=resMat[:maxProposalsIoU,:]
    gtLeft=numpy.matlib.repmat(gtMat[:,0],resMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],resMat.shape[0],1)
    gtRight=numpy.matlib.repmat((gtMat[:,0]+gtMat[:,2])-1,resMat.shape[0],1)
    gtBottom=numpy.matlib.repmat((gtMat[:,1]+gtMat[:,3])-1,resMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],resMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],resMat.shape[0],1)

    resLeft=numpy.matlib.repmat(resMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(resMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat((resMat[:,0]+resMat[:,2])-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat((resMat[:,1]+resMat[:,3])-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(resMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(resMat[:,3],gtMat.shape[0],1).T

    intL=np.max([resLeft,gtLeft],axis=0)
    intT=np.max([resTop,gtTop],axis=0)

    intR=np.min([resRight,gtRight],axis=0)
    intB=np.min([resBottom,gtBottom],axis=0)

    intW=(intR-intL)+1
    intW[intW<0]=0

    intH=(intB-intT)+1
    intH[intH<0]=0

    I=intH*intW
    U=resWidth*resHeight+gtWidth*gtHeight-I
    minsurf=np.minimum(resHeight*resWidth,gtWidth*gtHeight)
    IoU=I/(U+.0000000001)
    resTLRB=resMat[:,:]
    resTLRB[:,[2,3]]+=resTLRB[:,[0,1]]-1
    gtTLRB=gtMat[:,:]
    gtTLRB[:,[2,3]]+=gtTLRB[:,[0,1]]-1
    return (IoU,I,U,minsurf)

#filename conversions
def createRequiredDirs(filenameList,fromFilesDir):
    filesDirList=set(['/'.join(f.split('/')[:-1]) for f in filenameList])
    if fromFilesDir[0]=='+':
        for fd in filesDirList:
            addP=fromFilesDir[1:].split('/')
            addP[-1]=addP[-1]+fd.split('/')[-1]
            go('mkdir -p '+fd+'/'+'/'.join(addP))
    else:
        for fd in filesDirList:
            go('mkdir -p '+fd+'/'+fromFilesDir)


def getInputFromConf(confFname):
    return switches['inputDir']+confFname.split('/')[-1].split('.')[0]+'.png'


def getThresholdFromHm(hmFname,outDir):
    pathList=hmFname.split('/')
    pathList[-2]=outDir+pathList[-2]
    return '/'.join(pathList)[:-3]+'csv'


def getProposalFromConf(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]=switches['propDir']
    return ('/'.join(pathList)) [:-3]+'csv'


def getConfFromHm(hmFname):
    pathList=hmFname.split('/')
    pathList[-2]='conf_'+pathList[-2]
    return '/'.join(pathList)[:-3]+'csv'


def getIouFromConf(confFname):
    pathList=confFname.split('/')
    pathList[-2]='iou_'+pathList[-2]
    return '/'.join(pathList)[:-3]+'png'


def getGtFromConf(imageFname):
    pathList=imageFname.split('/')
    pathList[-2]='gt'
    return '/'.join(pathList)[:-3]+'txt'


def getProposalFromImage(imageFname,thr=0,hm=''):
    if thr==0:
        pathList=imageFname.split('/')
        pathList[-2]=switches['propDir']
        return '/'.join(pathList)[:-3]+'csv'
    else:
        pathList=imageFname.split('/')
        pathList[-2]='prop%d%s'%(int(thr*100),hm)
        return '/'.join(pathList)[:-3]+'csv'


def getProposalFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]=switches['propDir']
    return '/'.join(pathList)

def getConfidenseFromHeatmap(heatmapFname):
    if heatmapFname[-3:]=='png':
        heatmapFname=heatmapFname[:-3]+'csv'
    pathList=heatmapFname.split('/')
    pathList[-2]='conf_'+pathList[-2]
    return '/'.join(pathList)

getConfidenseFromProposal=getConfidenseFromHeatmap

#str 2 numpy
def arrayToCsvStr(arr):
    if len(arr.shape)!=2:
        raise Exception("only 2D arrays")
    resLines=[list(row) for row in list(arr)]
    return '\n'.join([','.join([str(col) for col in row])  for row in resLines])

def csvStr2Array(csvStr):
    return np.array([[float(c) for c in l.split(',')] for l in csvStr.split('\n') if len(l)>0])

def fname2Array(fname):
    if fname[-3:]=='png':
        return cv2.imread(fname,cv2.IMREAD_GRAYSCALE)/255.0
    else: #assuming csv
        if os.path.isfile(fname) and os.path.getsize(fname)>0:
            res= np.genfromtxt(fname, delimiter=',')
            return res
        else:
            return np.empty([0,5])


def array2csvFname(arr,csvFname):
    np.savetxt(csvFname,arr, '%9.5f',delimiter=',')

def array2pngFname(arr,pngFname):
    cv2.imwrite(pngFname,(arr*255).astype('uint8'),[cv2.IMWRITE_PNG_COMPRESSION ,0])


def loadTxtGtFile(fname):
    txt=myread(fname)
    lines=[l.strip().split(',') for l in txt.split('\n') if len(l.strip())>0]
    print '{%s}'%lines
    if len(lines)==0:
        return np.zeros([0,4]),np.zeros([0],dtype=object)

    for line in lines:
        if len(line) < 4:
            print 'BAD LINE:',str(line),'\nFile:',fname
            raise Exception()
            lines.remove(line)
        elif line == ['|']:
            lines.remove(line)

    for line in lines:
        if len(line) < 4:
            lines.remove(line)
        elif line == ['|']:
            lines.remove(line)

    if len(lines[0])>8:#4 points
        rects=np.empty([len(lines),4],dtype='float')
        tmpArr=np.array([[int(c) for c in line[:8]] for line in lines])
        left=tmpArr[:,[0,2,4,6]].min(axis=1)
        right=tmpArr[:,[0,2,4,6]].max(axis=1)
        top=tmpArr[:,[1,3,5,7]].min(axis=1)
        bottom=tmpArr[:,[1,3,5,7]].max(axis=1)
        rects[:,0]=left
        rects[:,1]=top
        rects[:,2]=1+right-left
        rects[:,3]=1+bottom-top
        trans=[','.join(line[8:]) for line in lines]
    else:#ltwh
        rects=np.array([[int(c) for c in line[:4]] for line in lines],dtype='float')
        trans=[','.join(line[4:]) for line in lines]
    return (rects,trans)


def loadVggTranscription(fname):
    txt=myread(fname)
    lines=[l.split(',') for l in txt.split('\n')]
    LTWHConf=np.empty([len(lines),5])
    transcriptions=np.empty([LTWHConf.shape[0]],dtype='object')
    transcriptions[:]=[l[-1] for l in lines]
    if len(lines) > 1:
    	LTWHConf[:,:]=np.array([[float(c) for c in l[:-1]] for l in lines])
    return LTWHConf,transcriptions

def getDontCare(transcriptions,dictionary=[]):
    dictionary=set(dictionary)
    if len(dictionary)==0:
        return np.array([tr!='###' for tr in transcriptions],dtype='bool')
    else:
        return np.array([(tr in dictionary) for tr in transcriptions],dtype='bool')


def getNmsIdx(LTWHConf):
    (IoU,I,U,minsurf)=get2PointIU(LTWHConf[:,:4],LTWHConf[:,:4])
    iouThr=eval(switches['iouThr'])
    return np.where((np.triu(IoU,1)>iouThr).sum(axis=0)<1)[0]
    #return LTWHConf[(np.triu(IoU,1)>iouThr).sum(axis=0)<1,:]


#algorithm
def getConfidenceForAll(hm,prop):
    prop=prop.astype('int32')
    ihm=np.zeros([hm.shape[0]+1,hm.shape[1]+1])
    ihm[1:,1:]=hm.cumsum(axis=0).cumsum(axis=1) #integral image
    confidenceDict=defaultdict(list)
    for rectId in range(prop.shape[0]):
        rect=tuple(prop[rectId,:4])
        (l,t,w,h)=rect
        r=l+w#the coordinates are translated by 1 because of ihm zero pad
        b=t+h
        confidenceDict[rect].append(((ihm[b,r]+ihm[t,l])-(ihm[b,l]+ihm[t,r]))/(w*h))
    res=np.array([tup[1]+(tup[0],) for tup in sorted([(max(confidenceDict[rec]),rec) for rec in confidenceDict.keys()],reverse=True)])
    return res


def postProcessProp(propCSV):
    lines=[tuple([float(c) for c in  l.split(',')]) for l in  propCSV.split('\n') if len(l)>0]
    rectDict=defaultdict(list)
    for l in lines:
        rectDict[l[:4]].append(l[4:])
    reslines=[]
    for r in rectDict.keys():
        reslines.append(r+max(rectDict[r]))
    if len(reslines):
        proposals=np.array(reslines)
        proposals=proposals[np.argsort(-proposals[:,4]),:]
        proposals=[[int(p[0]),int(p[1]),int(p[2]),int(p[3]),p[4],p[5],int(p[6]),int(p[7]),int(p[8]),int(p[9])] for p in proposals.tolist()]
        return '\n'.join([','.join([str(c) for c in p]) for p in proposals])
    else:
        return ''



switches={'maxProposalsIoU':'200000',#IoU over this are not computed #20000
'gpu':None,
'img2propPath':'./tmp/TextProposalsInitialSuppression/img2hierarchy',
'weakClassifier':'./tmp/TextProposalsInitialSuppression/trained_boost_groups.xml',
'proto':'/tmp/dictnet_vgg_deploy.prototxt',
'pretrained':'/tmp/dictnet_vgg.caffemodel',
'vocabulary':'/tmp/dictnet_vgg_labels.txt',
'dictnetThr':'0.004',
'vocDir':'voc_strong',
'iouThr':'0.3',
'minibatchSz':'100',
'threads':'1',
'dontCareDictFile':'',
'IoUThresholds':'[.5]',
'extraPlotDirs':'{".":"Confidence"}',
'care':'True', #If true dont cares are supressed
'bayesianFname':'/tmp/bayesian.cPickle',
'plotter':'plt.semilogx',
'thr':'0.1',
'plotfname':'plots.pdf',
'fixedProps':'1000',#,
'propDir':'proposals',
'inputDir':'input'
}



if __name__=='__main__':
    hlp="""
    hm2conf blabla/*/heatma.../*.csv
    hm2conf blabla/*/heatma.../*.png
    img2prop blabla/*/input/*.jpg
    """
    params=[(len(p)>0 and p[0]!='-',p) for p in sys.argv]
    sys.argv=[p[1] for p in params if p[0]]
    switches.update(dict([p[1][1:].split('=') for p in params if not p[0]]))
    print 'Threads',int(switches['threads'])
    if sys.argv[1]=='dictnet2final':
        #dnThrCount=0
        def worker(vggfname):
            outDir='finalDN%dIou%dVoc%s_%s'%((int(100*eval(switches['dictnetThr']))),(int(100*eval(switches['iouThr']))),switches['vocDir'].split('_')[-1],vggfname.split('/')[-2])
            ofname=vggfname.split('/')
            ofname[-2]=outDir
            ofname='/'.join(ofname)
            LTWHConf,transcriptions=loadVggTranscription(vggfname)

            #FIXING UNSORTED YOLO/TEXTBOXE BEGIN
            LTWHTextnes=fname2Array(vggfname.replace('vggtr_',''))
            if set([tuple(l) for l in LTWHTextnes[:,:4].astype('int32').tolist()])!=set([tuple(l) for l in LTWHConf[:,:4].astype('int32').tolist()]):
            #if set([tuple(l) for l in LTWHTextnes[0][:4].astype('int32').tolist()])!=set([tuple(l) for l in LTWHConf[0][:4].astype('int32').tolist()]):
                print "Count not fincd consistent conf "+vggfname
                raise Exception()
            sortdeterministic=lambda x: (x[:,0]+x[:,1]*(10^4)+x[:,0]*(10^8)+x[:,1]*(10^12)).argsort()
            LTWHTextnes=LTWHTextnes[sortdeterministic(LTWHTextnes),:]#Making the textnes follow a dterministic order
            idx=sortdeterministic(LTWHConf)#alligning vggtr with confidence rectangles
            transcriptions=transcriptions[idx]#alligning vggtr with confidence rectangles
            LTWHConf=LTWHConf[idx,:]#alligning vggtr with confidence rectangles
            #now the 4 first columns of LTWHConf and LTWHTextnes should be the same rectangles
            if LTWHTextnes[:,:4].astype('int32').tolist()!=LTWHConf[:,:4].astype('int32').tolist():
                print "Failed to allign rectangles "+vggfname
                raise Exception()
            textnesSortedIdx=np.argsort(-LTWHTextnes[:,4])
            LTWHConf=LTWHConf[textnesSortedIdx,:]
            transcriptions=transcriptions[textnesSortedIdx]
            #FIXING UNSORTED YOLO/TEXTBOXE END

            print vggfname.split('/')[-2].split('_')[-1]+'/'+vggfname.split('/')[-1],' Initial :',LTWHConf.shape[0],
            filterIdx=LTWHConf[:,4]>eval(switches['dictnetThr'])
            print ' dictnet>'+str(float(int(eval(switches['dictnetThr'])*100))/100)+' kills:',(filterIdx==0).sum(),
            LTWHConf=LTWHConf[filterIdx,:]
            transcriptions=transcriptions[filterIdx]
            #print LTWHConf[:,2:4].max()
            #print '\n\n#2\n','\n'.join(transcriptions.tolist())
            if switches['vocDir'] and LTWHConf.size>0:
                vocFname=ofname.split('/')
                vocFname[-2]=switches['vocDir']
                vocFname='/'.join(vocFname)[:-3]+'txt'
                voc=set([s.lower().strip() for s in myread(vocFname).split('\n')[:-1]])
                filterIdx=np.ones(transcriptions.shape[0],dtype='bool')
                for k in range(transcriptions.shape[0]):
                    filterIdx[k]=transcriptions[k].lower() in voc
                print ' VOC kills:', (filterIdx==0).sum(),
                LTWHConf=LTWHConf[filterIdx,:]
                transcriptions=transcriptions[filterIdx]
            res=[]
            filterIdx=getNmsIdx(LTWHConf)
            print 'NMS kills:', LTWHConf.shape[0]-(filterIdx).shape[0],' SURVIVED:',(filterIdx.shape[0])
            LTWHConf=LTWHConf[filterIdx,:]
            transcriptions=transcriptions[filterIdx]
            for k in range(transcriptions.shape[0]):
                resLine=','.join([str(int(c)) for c in LTWHConf[k,:4]])+','+transcriptions[k]
                #print 'MAX RESLINE:',resLine
                res.append(resLine)
            open(ofname,'w').write('\n'.join(res))
        outDirL= lambda x:'finalDN%dIou%dVoc%s_%s'%((int(100*eval(switches['dictnetThr']))),(int(100*eval(switches['iouThr']))),switches['vocDir'].split('_')[-1],x.split('/')[-2])
        [go('mkdir -p '+d) for d in  set('/'.join(f.split('/')[:-2]+[outDirL(f)]) for f in  sys.argv[2:])]
        if int(switches['threads'])<=1:
            [worker(f) for f in sys.argv[2:]]
        else:
            pool=Pool(int(switches['threads']))
            pool.map(worker,sys.argv[2:])
        sys.exit(0)


    if sys.argv[1]=='conf2dictnet':
        import caffe
        download('http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt',switches['proto'])
        download('http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel',switches['pretrained'])
        download('http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt',switches['vocabulary'])
        if switches['gpu']!=None:
            caffe.set_mode_gpu()
            caffe.set_device(eval(switches['gpu']))
        net=caffe.Classifier(switches['proto'],switches['pretrained'],image_dims=(32,100))

        labs=np.array(open(switches['vocabulary']).read().split('\n')[:-1])
        def spotwords(LTWH,img,net,labs):
            imgTensor=np.empty([LTWH.shape[0],1,32,100],dtype='uint8')
            resLines=[]
            minibatchSz=eval(switches['minibatchSz'])
            paddedSz=((LTWH.shape[0]/minibatchSz)+1)*minibatchSz
            imgTensor=np.zeros([paddedSz,1,32,100])
            classProb=np.zeros([paddedSz,88172])
            #print "Shape"
            #print LTWH.shape
            if LTWH.shape[0] != 0:
                LTWH[LTWH[:,2]<=3,2]=3#resizing to small rectangles
                LTWH[LTWH[:,3]<=3,3]=3

            for k in range(LTWH.shape[0]):
                patch=cv2.resize(img[LTWH[k,1]:LTWH[k,1]+LTWH[k,3],LTWH[k,0]:LTWH[k,0]+LTWH[k,2]],(100,32)).astype('float')
                patch=patch-patch.mean()
                patch=patch/patch.std()
                patch=patch*128
                imgTensor[k,0,:,:]=patch
            net.blobs['data'].reshape(minibatchSz,1,32,100)
            for k in range(paddedSz/minibatchSz):
                #print imgTensor[k*minibatchSz:(k+1)*minibatchSz,:,:,:].shape
                net.blobs['data'].data[...] = imgTensor[k*minibatchSz:(k+1)*minibatchSz,:,:,:]
                #res=net.forward_all(data=imgTensor[k*minibatchSz:(k+1)*minibatchSz,:,:,:])
                before=time.time()
                res=net.forward()
                print 'forward:',1000*(time.time()-before)
            #res=net.forward_all(data=imgTensor)#['prob'].reshape([-1])
                classProb[k*minibatchSz:(k+1)*minibatchSz,:]=res['prob'].reshape([minibatchSz,-1])
            wordIdx=classProb[:LTWH.shape[0],:].argmax(axis=1)
            wordProb=classProb[:LTWH.shape[0],:].max(axis=1)
            resLines=[]
            for k in range(LTWH.shape[0]):
                line=','.join([str(int(LTWH[k,0])),str(int(LTWH[k,1])),str(int(LTWH[k,2])),str(int(LTWH[k,3])),str(wordProb[k]),labs[wordIdx[k]]])
                resLines.append(line)
            return '\n'.join(resLines)
        t=time.time()
        for confFname in sys.argv[2:]:
            print 'BEGGINING ',confFname,
            img=cv2.imread(getInputFromConf(confFname),cv2.IMREAD_GRAYSCALE)
            print getInputFromConf(confFname)
            transcr=confFname.split('/')
            transcr[-2]='vggtr_'+transcr[-2]
            go('mkdir -p '+'/'.join(transcr[:-1]))
            transcr='/'.join(transcr)
            if not os.path.isfile(transcr):
                LTWH=np.array([[int(float(c)) for c in l.split(',')[:4]] for l in open(confFname).read().split('\n') if len(l)])
                res=spotwords(LTWH,img,net,labs)
                open(transcr,'w').write(res)
                print ' DONE ',len(res.split('\n')),' lines in ',int(1000*(time.time()-t)),' msec.'
            else:
                print ' AVOIDED RECUMPUATION FILE ALREADY there '
        sys.exit(0)

