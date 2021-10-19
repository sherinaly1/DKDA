%% load data
clear;clc;addpath(('C:\Users\SherinFathy\Desktop\wacv2016ICIP\wacv_code\LDA_tb'));
% load('wacv_features_original3.mat');% load dataset
% data=icip_features;
% maxac=0;
% for thre=[.6:.1:1 10:10:100]
load wacv2016ICIP_features;data=wacv;global applyLDA;global ev;nc=6;clear wacv;
%% set experiemntals parameters
LOSO=0;% 1: LOSO accuracy; 0: if you want 60-40 hold-out cross validation
byPoseOrByIntensity=1; % 1: by Pose; 0: by Intensity;
flag=3; % the type of feature you want to use (see below)
s=[2 4 8 16 2^5 2^6 2^7 200];% select number of feature selected for the 3D features
idx_s=7;% for the S, number of reduced features
pose='S' ;  % select the pose; 'F' for frontal 'S' for non-frontal ; 'A' for both frontal and non-frontal
level='L1' ; % select the level in case you want to work by level; L1: level 1; L2: level 2; L3 level 3;
applyLDA=0; % 0:NoReduction ; 1:LDA ; 2:KDA
ev=.9;% LDA threshold; %
% options2.t=thre;
options.t=.6;
options2.t=90;
%%
IdxDir=strcat(pwd,'\Idx\');%
resultsDir=strcat(pwd,'\results\',lower(pose),'\');
figDir=strcat(pwd,'\fig\');

%%
if byPoseOrByIntensity==1
    %% prepare the index of faces for specified pose
    if pose=='F'
        Idx=find(data.poses==pose&data.labels<7);
    elseif pose=='S'
        Idx=find(data.poses~='F'&data.labels<7);
    else
        Idx=find(data.labels<7);
    end
    % Preparing the Features and Instances Index
    featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',pose,'_ADGaGlbp_SS=',num2str(s(idx_s)),'_SS2=500'));
elseif byPoseOrByIntensity==0 % by intensity    
    IdxDir='.\Models\';
    if strcmp(level,'L1')
        Idx=find(data.levels==1&data.labels<7);% L1        
    elseif strcmp(level,'L2')
        Idx=find(data.levels==2&data.labels<7);% L2        
    elseif strcmp(level,'L3')
        Idx=find(data.levels==3&data.labels<7);% L3
    end
    
    % Preparing the Features and Instances Index
    %     featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',level,'_withADG_bestSS_',num2str(s(idx_s))));
    featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',pose,'_ADGaGlbp_SS=',num2str(s(idx_s)),'_SS2=500'));
end

% compute Y
Y=data.labels(Idx);

% compute subjects
subjects=data.subjects(Idx,:);

% Features Extraction

% Angles
f.FeaturesG=data.G(Idx,:);
% minval=min(f.FeaturesG); maxval=max(f.FeaturesG);
% minrep=repmat(minval,[size(f.FeaturesG,1) 1]);
% maxrep=repmat(maxval,[size(f.FeaturesG,1) 1]);
% f.FeaturesG=(f.FeaturesG-minrep)./(maxrep-minrep);

%allAngles
f.AllG=data.aG(Idx,:);

% ULBP
f.FeaturesFLBP=data.ULBP(Idx,:);
% minval=min(f.FeaturesFLBP); maxval=max(f.FeaturesFLBP);
% minrep=repmat(minval,[size(f.FeaturesFLBP,1) 1]);
% maxrep=repmat(maxval,[size(f.FeaturesFLBP,1) 1]);
% f.FeaturesFLBP=(f.FeaturesFLBP-minrep)./(maxrep-minrep);

f.FeaturesFLBP_pca=data.TLBP(Idx,:);
% minval=min(f.FeaturesFLBP_pca); maxval=max(f.FeaturesFLBP_pca);
% minrep=repmat(minval,[size(f.FeaturesFLBP_pca,1) 1]);
% maxrep=repmat(maxval,[size(f.FeaturesFLBP_pca,1) 1]);
% f.FeaturesFLBP_pca=(f.FeaturesFLBP_pca-minrep)./(maxrep-minrep);

% reduced
f.FeaturesGr=data.G(Idx,unique(featureIdx.G_best_mc));
f.AllGr=data.aG(Idx,unique(featureIdx.aG_best_mc));
% minval=min(f.FeaturesGr); maxval=max(f.FeaturesGr);
% minrep=repmat(minval,[size(f.FeaturesGr,1) 1]);
% maxrep=repmat(maxval,[size(f.FeaturesGr,1) 1]);
% f.FeaturesGr=(f.FeaturesGr-minrep)./(maxrep-minrep);

FeaturesName=[];

%---------------------------------- 2D --------------------------------
%---------------------------------------------------------------------
if flag==1
    X=f.FeaturesFLBP;FeaturesName{flag}='ULBP';
elseif flag==2
    X=f.FeaturesFLBP_pca; FeaturesName{flag}='TLBP';
    %---------------------------------- 3D --------------------------------
    %---------------------------------------------------------------------
elseif flag==3
    X=f.FeaturesG;FeaturesName{flag}='G';
elseif flag==4
    X=f.AllG;FeaturesName{flag}='aG';
    elseif flag==5
    X=f.FeaturesGr;FeaturesName{flag}=['Gr',num2str(s(idx_s))];
elseif flag==6
      X=f.AllGr;FeaturesName{flag}='aGr';
    %---------------------------------- 2D + 3D ---------------------------
    %---------------------------------------------------------------------
elseif flag==7
    X=[f.FeaturesFLBP ];FeaturesName{flag}=['ULBP_G'];
    X2=f.FeaturesG;
elseif flag==8
    X=[f.FeaturesFLBP ];FeaturesName{flag}=['ULBP_aG'];
    X2=f.AllG;
elseif flag==9
    X=[f.FeaturesFLBP ];FeaturesName{flag}=['ULBP_Gr',num2str(s(idx_s))];
    X2=f.FeaturesGr;
elseif flag==10
    X=[f.FeaturesFLBP ];FeaturesName{flag}=['ULBP_aGr',num2str(s(idx_s))];
    X2=f.AllGr;
elseif flag==11
    X=[f.FeaturesFLBP_pca ];X2=f.FeaturesG;FeaturesName{flag}=['TLBP_G'];
elseif flag==12
    X=[f.FeaturesFLBP_pca ];X2=f.FeaturesG;FeaturesName{flag}=['TLBP_aG'];
elseif flag==13
    X=[f.FeaturesFLBP_pca];X2=[ f.FeaturesGr];FeaturesName{flag}=['TLBP_Gr',num2str(s(idx_s))];
elseif flag==14
     X=[f.FeaturesFLBP_pca];X2=[ f.AllGr];FeaturesName{flag}=['TLBP_aGr',num2str(s(idx_s))];
end
disp( sprintf( '... flag=%d %s - %s',flag,FeaturesName{flag},pose));

% compute subkect list
subjList=unique(subjects);
clear data;


%% Classification
% [acc_all,accPerSubj,computedLabel]=classificationPerSubject(DS);
if flag>6
    DS=composeDS(X,Y,subjects,FeaturesName{flag});
    DS2=composeDS(X2,Y,subjects,FeaturesName{flag});
    if byPoseOrByIntensity==1
        DS.name=strcat(DS.name,'_',pose);
    else
        DS.name=strcat(DS.name,'_',level);
    end
    clear X X2 f;
    if LOSO==0
        [acc_all,accPerSubj,computedLabel,trueLab,DS]=classification6040_2(DS,DS2,options,options2);
    else
        [acc_all,accPerSubj,computedLabel, trueLab,DS]=classificationPerSubject_2(DS,DS2,options,options2);
    end
else
    DS=composeDS(X,Y,subjects,FeaturesName{flag});
    if byPoseOrByIntensity==1
        DS.name=strcat(DS.name,'_',pose);
    else
        DS.name=strcat(DS.name,'_',level);
    end
    clear X f;
    if LOSO==0
        [acc_all,accPerSubj,computedLabel,trueLab,DS]=classification6040(DS,options);
    else
        [acc_all,accPerSubj,computedLabel, DS]=classificationPerSubject_1(DS,options);
    end
end
disp(sprintf('%s: %f . Maximum %f',DS.name,acc_all,max(accPerSubj)));
% bar(accPerSubj);
%% save results
varname = genvarname(strcat('acc',DS.name));
name=DS.name;trueLabels=DS.output';ss=s(idx_s);
save ([resultsDir,varname,'.mat'],'acc_all','computedLabel','name','trueLabels','ss','pose','applyLDA','ev');
% if maxac<acc_all
%     maxac=acc_all;
%     thrr=thre;
% end 
% end 
% disp(sprintf('Max for %s: is %f with threshold %f',DS.name,maxac,thrr));