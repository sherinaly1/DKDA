%% no reduction, LDA, DKDA,
%% applyKDA= 0,1,..3 when the thre1 and thre2 are already optimized in auto2 or auto 3
%%
clear;clc;addpath(('.\LDA_tb'));global applyLDA;global ev;
thresholds=[0.9	0;  ...ULBP
    10	0;  ...TLBP
    .9	0; ...HOG
    .9	0; ...GIST
    1   0;...LBP_shan
    
    100	0;  ...G
    100	0;  ...Gr
    0.6	0;  ... D
    0.6	0;  ... Dr
    .8   0;  ... mCurv
%     0 0;  ... mCurv_r
    
    0.9	80; ...ULBP_G
    0.9	80; ...ULBP_Gr128
    0.9	0.6; ...ULBP_D
    0.9	0.6; ...ULBP_Dr128
    0.9	0.8; ...ULBP_mCurv
    
    10	100;...TLBP+G
    10	100;...TLBP+Gr128
    10	0.6;...TLBP+D
    10	0.6;...TLBP+Dr128
    10	0.8;...TLBP+mCurv
    
    .9	100;...HOG+G
    .9	100;...HOG+Gr128
    .9	0.6;...HOG+D
    .9	0.6;...HOG+Dr128
    .9	0.8;...HOG+mCurv
    
    .9	100;...Gist+G
    .9	100;...Gist+Gr128
    .9	0.6;...Gist+D
    .9	0.6;...Gist+Dr128
    .9	0.8;...Gist+mCurv
    
    01	100;...LBP_shan+G
    01	100;...LBP_shan+Gr128
    01	0.6;...LBP_shan+D
    01	0.6;...LBP_shan+Dr128
    01  0.8;...LBP_shan+mCurv
    ];acc_all_=[];
for applyLDA=[2] % 0:NoReduction ; 1:LDA ; 2:KDA ;
    for poselevel=2:4
        load wacv2016ICIP_features_round2;data=wacv;nc=6;clear wacv;
        %% set experiemntals parameters
        LOSO=2;%0: if you want 60-40 hold-out cross validation; 1: LOSO accuracy; 2:perSequence
        modality=4; % 1:2D; 2:3D ; 3:2D+3D
        flag2d=[3 5];% 1:5
        flag3d=[2 4];% 1:5
        s=[2 4 8 16 2^5 2^6 2^7 200];% select number of feature selected for the 3D features
        idx_s=7;% for the S, number of reduced features
        if poselevel==1
            byPoseOrByIntensity=1; % 1: by Pose; 0: by Intensity;
            pose='F' ;  % select the pose; 'F' for frontal 'S' for non-frontal ; 'A' for both frontal and non-frontal
        elseif poselevel==2
            byPoseOrByIntensity=1; % 1: by Pose; 0: by Intensity;
            pose='S' ;
        elseif poselevel==3
            byPoseOrByIntensity=0; % 1: by Pose; 0: by Intensity;
            level='L1' ;
        else
            byPoseOrByIntensity=0;
            level='L2' ; % select the level in case you want to work by level; L1: level 1; L2: level 2; L3 level 3;
        end
        ev=.9;% LDA threshold; %
        if byPoseOrByIntensity==1
            options.tsPerc=.3;
        else
            options.tsPerc=.3;
        end
        if modality==1
            options.t=thresholds(flag2d,1);
        elseif modality==2
            options.t=thresholds(5+flag3d,1);
        elseif modality ==3
            options.t=thresholds(flag2d,1);
            options2.t=thresholds(10+(5*(flag2d-1)+flag3d),2);
        else% NKDA
            alloptions=[];
            for fl=1:length(flag2d)
                options.t=thresholds(flag2d(fl),1);
                alloptions=[alloptions;{options}];
            end
            for fl=1:length(flag3d)
                options2.t=thresholds(5+flag3d(fl),1);
                alloptions=[alloptions;{options2}];
            end
        end
        %%
        IdxDir=strcat(pwd,'\Idx\');%
        resultsDir=strcat(pwd,'\results\');
        figDir=strcat(pwd,'\fig\');
        
        %%
        if byPoseOrByIntensity==1
            %% prepare the index of faces for specified pose
            if pose=='F'
                Idx=find(data.poses==pose&data.labels<7);
            elseif pose=='R'
                Idx=find(data.poses=='R'&data.labels<7);
            elseif pose=='L'
                Idx=find(data.poses=='L'&data.labels<7);
            elseif pose=='S'
                Idx=find(data.poses~='F'&data.labels<7);
            else
                Idx=find(data.labels<7);
            end
            % Preparing the Features and Instances Index
            featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',pose,'_ADGaGlbp_SS=',num2str(s(idx_s)),'_SS2=500'));
            featureIdx_c=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',pose,'_Curcature_SS=',num2str(64),'_SS2=500'));
        elseif byPoseOrByIntensity==0 % by intensity
            
            if strcmp(level,'L1')
                Idx=find(data.levels==1&data.poses=='F'&data.labels<7);% L1
            elseif strcmp(level,'L2')
                Idx=find(data.levels==2&data.poses=='F'&data.labels<7);% L2
            elseif strcmp(level,'L3')
                Idx=find(data.levels==3&data.poses=='F'&data.labels<7);% L3
            end
            % Preparing the Features and Instances Index
            %     featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',level,'_withADG_bestSS_',num2str(s(idx_s))));
            featureIdx=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',level,'_ADGaGlbp_SS=',num2str(s(idx_s)),'_SS2=500'));
            featureIdx_c=load (strcat(IdxDir,'featuresReducedIdx_mc_',num2str(nc),'c_',level,'_Curcature_SS=',num2str(64),'_SS2=500'));
        end
        
        % compute Y
        Y=data.labels(Idx);
        sequence=data.sequence(Idx,1);
        subjects=data.subjects(Idx,:);
        
        % Features Extraction
        
        f2DList=[  {data.ULBP(Idx,:)};...  % ULBP
            {data.TLBP(Idx,:)};...         % TLBP
            {data.hog_Dalal(Idx,:)};...    % HOG
            {data.gist(Idx,:)}; ...        % Gist
            {data.lbp_shan(Idx,:)}; ...   % lbp_shan
            ];
        f2DName=[  {'ULBP'};...  % ULBP
            {'TLBP'};...         % TLBP
            {'HOG'};...     % HOG
            {'Gist'}; ...        %Gist
            {'LBP_shan'}; ...    % lbp_shan
            ];
        f3DList=[{data.G(Idx,:)}; ...                    % G
            {data.G(Idx,unique(featureIdx.G_best_mc))}; ...  % Gr
            {data.allD(Idx,:,10)}; ...                             % D
            {data.allD(Idx,unique(featureIdx.D_best_mc),10)};      % Dr
            {data.Cmean(Idx,:)};...                     % Curvature
            {data.Cmean(Idx,unique(featureIdx_c.C_best_mc))};... % curvature_r
            ];
        f3DName=[{'G'}; ...  % G
            {'Gr'}; ...  % Gr
            {'D'}; ...    % D
            {'Dr'};      % Dr
            {'mCurv'};...  % mCurvature
            {'mCurv_r'};...  % mCurvature_r
            ];
        
        %     if byPoseOrByIntensity==1
        %         if modality==
        %         disp( sprintf( '...%s - %s - applyLDA=%d',FeaturesName{flag},pose,applyLDA));
        %     else
        %         disp( sprintf( '...%s - %s- applyLDA=%d',FeaturesName{flag},level,applyLDA));
        %
        %     end
        % compute subject list
        subjList=unique(subjects);
        clear data;
        
        %% Classification
        if modality==4
            for fl=1:length(flag2d)
            DS=composeDS(f2DList{flag2d(fl)},Y,subjects,f2DName{flag2d(fl)},sequence);
            DSList{fl}=DS;
            end
            for fl=1:length(flag3d)
            DS2=composeDS(f3DList{flag3d(fl)},Y,subjects,f3DName{flag3d(fl)});
            DSList2{fl}=DS2;
            end
%             if byPoseOrByIntensity==1
%                 DS.name=strcat(DS.name,'_',DS2.name,'_',pose);
%             else
%                 DS.name=strcat(DS.name,'_',DS2.name,'_',level);
%             end
            clear f2DList f3DList f DS DS2;
            if LOSO==0
%                 [acc_all,accPerSubj,computedLabel,trueLab,DS]=classification6040_2(DS,DS2,options,options2);
            elseif LOSO==1
%                 [acc_all,accPerSubj,computedLabel, trueLab,DS]=classificationPerSubject_2(DS,DS2,options,options2);
            else
                [acc_all,accPerSubj,computedLabel, trueLab,DS]=classificationPerSequence_3(DSList,DSList2,alloptions);
            end
        elseif modality==3
            DS=composeDS(f2DList{flag2d},Y,subjects,f2DName{flag2d},sequence);
            DS2=composeDS(f3DList{flag3d},Y,subjects,f3DName{flag3d});
            if byPoseOrByIntensity==1
                DS.name=strcat(DS.name,'_',DS2.name,'_',pose);
            else
                DS.name=strcat(DS.name,'_',DS2.name,'_',level);
            end
            clear f2DList f3DList f;
            if LOSO==0
                [acc_all,accPerSubj,computedLabel,trueLab,DS]=classification6040_2(DS,DS2,options,options2);
            elseif LOSO==1
                [acc_all,accPerSubj,computedLabel, trueLab,DS]=classificationPerSubject_2(DS,DS2,options,options2);
            else
                [acc_all,accPerSubj,computedLabel, trueLab,DS]=classificationPerSequence_2(DS,DS2,options,options2);
            end
        else
            if modality==1
                DS=composeDS(f2DList{flag2d},Y,subjects,f2DName{flag2d},sequence);
            else
                DS=composeDS(f3DList{flag3d},Y,subjects,f3DName{flag3d},sequence);
            end
            if byPoseOrByIntensity==1
                DS.name=strcat(DS.name,'_',pose);
            else
                DS.name=strcat(DS.name,'_',level);
            end
            clear f2DList f3DList f;
            if LOSO==0
                [acc_all,accPerSubj,computedLabel,trueLab,DS]=classification6040(DS,options);
            elseif LOSO==1
                [acc_all,accPerSubj,computedLabel, DS]=classificationPerSubject_1(DS,options);
            elseif LOSO==2
                [acc_all,accPerSubj,computedLabel,trueLab,DS]=classificationPerSequence_1(DS,options);
            end
        end
        disp(sprintf(' HOG+LBP+Gr+D: %f options.tsPerc=%f. Maximum %f',acc_all,options.tsPerc,max(accPerSubj)));
        % bar(accPerSubj);
        %% save results
%         varname = genvarname(strcat('acc',DS.name));
%         name=DS.name;trueLabels=DS.output';ss=s(idx_s);
%         if byPoseOrByIntensity==0
%             pose=level;
%         end
%         save ([resultsDir,varname,'.mat'],'acc_all','computedLabel','name','trueLabels','ss','pose','applyLDA','ev');
        % end
        % disp(sprintf('Max for %s: is %f with threshold %f',DS.name,maxac,thrr));
        acc_all_(applyLDA+1,poselevel)=acc_all
    end
end