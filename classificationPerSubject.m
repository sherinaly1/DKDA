function [acc_all,accPerSubj,computedLabel]=classificationPerSubject(DS)
if size(DS.output,2)>1
    X=DS.input';Y=DS.output';subjects=DS.subjects;
else
    X=DS.input;Y=DS.output;subjects=DS.subjects;
end
computedLabel=[];accPerSubj=0;
subjList=unique(subjects);
for setno = 1:length(subjList)
     disp( sprintf( ...
            ' ...classificationPerSubject for subject %s: %s  ,mean=%d --  setno = %d of %d --  ', ...
            subjList{setno},DS.name,round(mean(accPerSubj)),setno,length(subjList)));
    trIdx = find(~strcmp(subjects,subjList{setno}));
    tsIdx =  find(strcmp(subjects,subjList{setno}));
    %         acc(setno)=mysvm_5RBF_mc(X(trIdx,:),y(trIdx,1), X(tsIdx,:),y(tsIdx,1))
    [accPerSubj(setno),y_]=mysvm_5L_mc(X(trIdx,:),Y(trIdx,1), X(tsIdx,:),Y(tsIdx,1));
    computedLabel(tsIdx,1)=y_;
end
acc_all=sum(computedLabel==Y)/length(Y);
