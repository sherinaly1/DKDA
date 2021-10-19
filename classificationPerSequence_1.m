function [acc_all,accPerSubj,computedLabel,trueLab,DS]=classificationPerSequence_1(DS,options)
global applyLDA;
% global ev;
if size(DS.output,2)>1
    X=DS.input';Y=DS.output';sequence=DS.sequence';
else
    X=DS.input;Y=DS.output;
end

if applyLDA==1
    %%
    DS.name=strcat(DS.name,'_LDA');
elseif applyLDA==2
    DS.name=strcat(DS.name,'_KDA_Guss_',num2str(options.t));
elseif applyLDA==6
    DS.name=strcat(DS.name,'_KSR_Guss_',num2str(options.t));
elseif applyLDA==4
    DS.name=strcat(DS.name,'_LSDA');
elseif applyLDA==5
    DS.name=strcat(DS.name,'_SRKDA_',num2str(options.t));
end
computedLabel=[];accPerSubj=0;
seed_=[0 55 333 653 1035];
seqno=unique(sequence);
for setno = 1:5
    rng('default')
    rng(seed_(setno));
    computedLabel=[];
    rndmIdx=randi([1 length(seqno)],round(length(seqno)*options.tsPerc),1);
    tsSeqNo = sort(seqno(rndmIdx));
    trSeqNo=setdiff(seqno,tsSeqNo);
    trIdx = ismember(sequence,trSeqNo);
    tsIdx = ismember(sequence,tsSeqNo);[sum(trIdx) sum(tsIdx)]
    
    if applyLDA==1
        %%
        options_ = [];
        options_.Fisherface = 1;
        [eigvector, ~] = CaiLDA(Y(trIdx), options_, X(trIdx,:));
        X_tr = X(trIdx,:)*eigvector;
        X_ts=X(tsIdx,:)*eigvector;
        %%
        disp( sprintf( ...
            ' ...classificationPerSequence: %s tsPerc=%f ,mean=%d --  setno = %d of 5 -- Dim=%d ', ...
            DS.name,options.tsPerc,round(mean(accPerSubj)),setno,size(X_tr,2)));
        %         acc(setno)=mysvm_5RBF_mc(X(trIdx,:),y(trIdx,1), X(tsIdx,:),y(tsIdx,1))
        [accPerSubj(setno),y_]=mysvm_5L_mc(X_tr,Y(trIdx),X_ts,Y(tsIdx));
    elseif applyLDA==2
        fea=X(trIdx,:);gnd=Y(trIdx);
        options.KernelType = 'Gaussian';
        % options.t = options.t;
        [eigvector, ~] = KDA(options,gnd,fea);
        
        feaTest = X(tsIdx,:);
        Ktest = constructKernel(feaTest,X(trIdx,:),options);
        X_ts= Ktest*eigvector;
        Ktrain = constructKernel(X(trIdx,:),X(trIdx,:),options);
        X_tr= Ktrain*eigvector;
        disp( sprintf( ...
            ' ...classificationPerSequence: %s tsPerc=%f ,mean=%d --  setno = %d of 5 --  t=%d -- Dim=%d', ...
            DS.name,options.tsPerc,round(mean(accPerSubj)),setno,options.t,size(X_tr,2)));
        Y_tr=Y(trIdx);Y_ts=Y(tsIdx);
        [accPerSubj(setno),y_]=mysvm_5L_mc(X_tr,Y_tr,X_ts,Y_ts);
    elseif applyLDA==3%KSR
        
        fea=X(trIdx,:);gnd=Y(trIdx);
        options.gnd = gnd;
        options.ReguAlpha = 0.01;
        options.ReguType = 'Ridge';
        options.KernelType = 'Gaussian';
        Ktrain = constructKernel(fea,[],options);
        options.Kernel = 1;
        [eigvector] = KSR_caller(options, Ktrain);
        
        X_tr = Ktrain*eigvector;    % X_tr is training samples in the KSR subspace
        
        feaTest = X(tsIdx,:);
        Ktest = constructKernel(feaTest,fea,options);
        X_ts = Ktest*eigvector;    % X_ts is test samples in the KSR subspace
        disp( sprintf( ...
            ' ...classificationPerSequence: %s  ,mean=%d --  setno = %d of 5 --  t=%d', ...
            DS.name,round(mean(accPerSubj)),setno,options.t));
        Y_tr=Y(trIdx);Y_ts=Y(tsIdx);
        [accPerSubj(setno),y_]=mysvm_5L_mc(X_tr,Y_tr,X_ts,Y_ts);
    elseif applyLDA==4
        fea=X(trIdx,:);gnd=Y(trIdx);
        options.beta=.5;
        options.k = 1;
        [eigvector, ~] = LSDA(gnd, options, fea);
        X_tr = X(trIdx,:)*eigvector;
        X_ts=X(tsIdx,:)*eigvector;
        [accPerSubj(setno),y_]=mysvm_5L_mc(X_tr,Y(trIdx),X_ts,Y(tsIdx));
    elseif applyLDA==5
        fea=[X(trIdx,:)] ;gnd=Y(trIdx);
        feaTest=[X(tsIdx,:)];
        fea = NormalizeFea(fea);   
        feaTest=NormalizeFea(feaTest);
        opt = [];
        opt.KernelType = 'Gaussian';
        opt.t = options.t;
        opt.ReguAlpha = 0.001;
        model = SRKDAtrain(fea, gnd, opt);
        [accc ,y_] = SRKDApredict(feaTest, Y(tsIdx), model); accPerSubj(setno)=accc*100;
    else
        disp( sprintf( ...
            ' ...classificationPerSequence: %s tsPerc=%f ,mean=%d --  setno = %d of 5 -- Dim=%d ', ...
            DS.name,options.tsPerc,round(mean(accPerSubj)),setno,size(X(trIdx,:),2)));
        %         acc(setno)=mysvm_5RBF_mc(X(trIdx,:),y(trIdx,1), X(tsIdx,:),y(tsIdx,1))
        [accPerSubj(setno),y_]=mysvm_5L_mc(X(trIdx,:),Y(trIdx,1), X(tsIdx,:),Y(tsIdx,1));
    end
    
    computedLabel(:,1)=y_;trueLab=Y(tsIdx);
end
acc_all=mean(accPerSubj);%sum(computedLabel==Y(tsIdx))/length(Y(tsIdx));
