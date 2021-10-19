load wacv2016ICIP_features;data=wacv;nc=6;clear wacv;
%%
% neutralIdx=find(wacv.labels==7);
% for i=1:length(neutralIdx)-1
%     sequence(neutralIdx(i):neutralIdx(i+1),1)=i;
% end
% i=i+1;
% sequence(neutralIdx(i):length(wacv.labels),1)=i;
% wacv.sequence=sequence;
% save wacv2016ICIP_features wacv
%%
seqno=unique(data.sequence);
seed_=[0 55 333 653 1035];
for setno=1:length(seed_)
    rng('default')
    rng(seed_(setno));
    trSeqNo = randi([min(seqno) max(seqno)],round(length(seqno)*.6),1);
    tsSeqNo=setdiff(seqno,trSeqNo);
    trIdx = ismember(data.sequence,trSeqNo);
    tsIdx = ismember(data.sequence,tsSeqNo);
end
