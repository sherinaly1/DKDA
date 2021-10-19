function featureSelection()
close all;clc;clear;
global plotAndSav c1 c2  nc ss plotFlag labels feature featureName
%%
addpath(genpath('F:\wacv2016ICIP\wacv2016_code_round2\ToolboxBalu3'));
%%
load wacv2016ICIP_features_round2_reduced;data=wacv;nc=6;clear wacv;
%% parameters
plotAndSav=1;poseFlag=1;nc=6;  pose='L2';
s3d=[ 2^6];
s2d=[100 200 300 400 500 600 700 800];
Classes = {'Happy','Surprise','Sad','Anger','Disgust','Fear','Neutral'};
dir=[pwd,'\ReducedFeatures\'];
figDir=[pwd,'\ReducedFeatures\fig\'];
% loop=5;
%%
% Initialization
for s1=1:length(s3d)
    for s2=5%1:length(s2d)
        ss=s3d(s1);ss2=s2d(s2);
        C_mc=[];C_best_mc=[];
        
        for c1=1:nc-1
            for c2=c1+1:nc
                
                
                if strcmp(pose,'F')
                    Idx=find(data.poses==pose&data.labels<7);
                elseif strcmp(pose,'S')
                    Idx=find(data.poses~='F'&data.labels<7);
                elseif strcmp(pose,'A')
                    Idx=find(data.labels<7);
                    
                elseif strcmp(pose,'L1')
                    Idx=find(data.levels==1&data.poses=='F'&data.labels<7);% L1
                elseif strcmp(pose,'L2')
                    Idx=find(data.levels==2&data.poses=='F'&data.labels<7);% L2
                elseif strcmp(pose,'L3')
                    Idx=find(data.levels==3&data.poses=='F'&data.labels<7);% L3
                end
                
                % read distances
                curvature=data.Cmean(Idx,:);
                
                % read labels
                labels=data.labels(Idx);
                %  faces=data.face3d(Idx,:);
                C=[];
                %                 %% Curvature
                %                 % ============
                %                 feature=curvature;featureName='Curvature'; plotFlag=2;
                %                 [d]=SelectTheBest(ss);
                %                 C=[C;d];
                %                 C_mc=[C_mc;d];
                
                %% Angles
                %=====================
                feature=angles;  plotFlag=3; featureName='Angles';
                [d]=SelectTheBest(ss);
                G=[G;d];
                G_mc=[G_mc;d];
                %
                %                 %% All Angles
                %                 %=====================
                %                 % select the best 10 Disctances, Rank (roc)
                %                 feature=allAngles; featureName='All_Angles';plotFlag=5;
                %                 [d]=SelectTheBest(ss);
                %                 aG=[aG;d];
                %                 aG_mc=[aG_mc;d];
                %
                %                 %% Areas
                %                 %=====================
                %                 feature=areas;plotFlag=4; featureName='Areas';
                %                 [d]=SelectTheBest(ss);
                %                 A=[A;d];
                %                 A_mc=[A_mc;d];
                %
                %
                %% generate best D, A, G, aG, lbp
                DD=C(:);
                flp=tabulate(DD);flp=flp(flp(:,2)~=0,[2,1]);
                ff=sortrows(flp,1);
                C_best=ff(end:-1:end-ss+1,2);
                C_best_mc=unique([C_best_mc;ff(end:-1:end-ss+1,2)]);
                
                
                %   plotSelectedFeatures_forSubPlot_definedLineColores(faces,D_best,2,'');
                %             savePlot(figDir,strcat('All_Distance_s=',num2str(ss),'_',num2str(c1),'_',num2str(c2),'_',pose));
                
                save (strcat(dir,'featuresReducedIdx_Curvature_',num2str(nc),'c_',num2str(c1),'_',num2str(c2),'_',pose,'_SS=',num2str(ss),'_SS2=',num2str(ss2)),...
                    'C', 'C_best') ;
                
            end
        end
        % plotSelectedFeatures_forSubPlot_definedLineColores(faces,D_best_mc,2,'');
        % savePlot(figDir,strcat('AllClasses_6c_Distance_s=',num2str(ss),'_',pose));
        
        save (strcat(dir,'featuresReducedIdx_mc_',num2str(nc),'c_',pose,'_Curcature_SS=',num2str(ss),'_SS2=',num2str(ss2)),...
            'C_mc', 'C_best_mc') ;
        % end
    end
end
end
function [D]=SelectTheBest(ss)
global criteria
D=[];
criteria='roc';
[allIdx]=applyCriteria(ss);
D=[D allIdx];
plotSelectedFeature(allIdx);

criteria='entropy';
[allIdx]=applyCriteria(ss);
D=[D allIdx];
plotSelectedFeature(allIdx);

criteria='ttest';
[allIdx]=applyCriteria(ss);
D=[D allIdx];
plotSelectedFeature(allIdx);

criteria= 'brattacharyya';
[allIdx]=applyCriteria(ss);
D=[D allIdx];
plotSelectedFeature(allIdx);

criteria = 'wilcoxon';
[allIdx]=applyCriteria(ss);
D=[D allIdx];
plotSelectedFeature(allIdx);
end
function [allIdx]=applyCriteria(ss)
global criteria feature labels
op=[];
op.m = ss;                     % 10 features will be selected
op.criterion = criteria;          % ROC criterion will be used
op.show = 1;                   % display results
[allIdx] = Bfs_rank(feature,labels,op);
end
function plotSelectedFeature(idx)
global plotAndSav plotFlag %c1 c2 criteria nc ss   featureName
if plotAndSav
    plotThisFace(idx,plotFlag,'');
    %     savePlot(figDir,strcat(criteria,'_',featureName,'_',num2str(c1),'_',num2str(c2),'_',num2str(nc),'c_',num2str(ss)));
end
end
function plotThisFace(s,flag,titl,colored)

close all;
if ~exist('colored','var')
    colored=0;
end
load KinectE;E=KinectE;
load KinectTri;angelsList=KinectTri;
plotData=load('toPlotFaceModel');
addpath(genpath([pwd,'\toolbox_graph\toolbox_graph']));
%face=reshape(faces(1,:),[3 121]); face=face';%% i commented this to let the
%shapes be plotted using the plot mesh and use the same face of it saved in
%plotData.vertex

face=plotData.vertex';
%% plotting selected features
% plot3(face(:,1),face(:,2),face(:,3),'.r');hold on
leftE=[3:4,14:17,19,21,24,28,30,36,32,34,38,43,45,48,50,52,149:151,154,156:270,273:275,278:280,283:284,288:290,293:295,297,299,301,303,305,307,308:313];
rightE=setdiff(1:size(E,1),leftE);
% for j=right%1:size(E,1)
%     j
%     text(face(E(j,:),1), face(E(j,:),2), face(E(j,:),3),num2str(j));
%     line(face(E(j,:),1), face(E(j,:),2), face(E(j,:),3));hold on;
% end

%   load KinctTri;angelsList=KinctTri;
%
leftT=[3,5,9,10,17:22,31:38,42:44,48:50,52,54,55,61:65,70,71,76,...
    77:79,81,84:85 ,93:99,103:105,109:111,133:152,155:158,161,162,165:174,...
    185:192,203,204,206];
rightT=setdiff(1:size(angelsList,1),leftT);
%     nlines=size(angelsList,1);
%     cc=hsv(nlines);
%     for j=left%1:size(angelsList,1)
%         j
%         fill3(face(angelsList(j,:),1),  face(angelsList(j,:),2), face(angelsList(j,:),3),cc(j,:));hold on;
%        text(face(angelsList(j,1),1),  face(angelsList(j,1),2), face(angelsList(j,1),3)-0.02, num2str(j), 'Color', cc(j,:),'FontSize',14);hold on
%         %         pause
%     end
%     hold off;

plot_mesh(plotData.vertex,plotData.faces, plotData.options);
if colored
    colormap jet(256);
    % else
    %     colormap gray;
end
hold on;
if flag==1% plot keypoints
    for i=1:length(s)
        plot3(face(s(i),1),face(s(i),2),face(s(i),3), 'r*','LineWidth',3);hold on
    end
    hold off;
elseif flag==2% plot distancs
    nlines=length(s);
    cc=hsv(size(E,1));
    for j=1:length(s)
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(E(s(j),:),1), face(E(s(j),:),2), face(E(s(j),:),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
        % plot3(face(angelsList(j,2:3),1),  face(angelsList(j,2:3),2), face(angelsList(j,2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %   text(face(E(s(j),1),1), face(E(s(j),1),2), face(E(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        % pause
    end
    hold off
elseif flag==22
    nlines=length(s);
    cc=hsv(nlines);
    for j=1:length(s)
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(E(rightE(s(j)),:),1), face(E(rightE(s(j)),:),2), face(E(rightE(s(j)),:),3),'Color',cc(j,:),'LineWidth',3);hold on;
        % plot3(face(angelsList(j,2:3),1),  face(angelsList(j,2:3),2), face(angelsList(j,2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %   text(face(E(s(j),1),1), face(E(s(j),1),2), face(E(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        % pause
    end
    hold off
elseif flag==23
    nlines=length(s);
    cc=hsv(nlines);
    for j=1:length(s)
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(E(leftE(s(j)),:),1), face(E(leftE(s(j)),:),2), face(E(leftE(s(j)),:),3),'Color',cc(j,:),'LineWidth',3);hold on;
        % plot3(face(angelsList(j,2:3),1),  face(angelsList(j,2:3),2), face(angelsList(j,2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %   text(face(E(s(j),1),1), face(E(s(j),1),2), face(E(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        % pause
    end
    hold off
elseif flag==3%plot angles
    %     load KinctTri;angelsList=KinctTri;
    nlines=length(s);
    
    cc=hsv(size(angelsList,1));
    for j=1:length(s)
        
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(angelsList(s(j),1:2),1), face(angelsList(s(j),1:2),2), face(angelsList(s(j),1:2),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
        plot3(face(angelsList(s(j),2:3),1),  face(angelsList(s(j),2:3),2), face(angelsList(s(j),2:3),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
        %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        %         pause
        
    end
    hold off;
elseif flag==32
    nlines=length(s);
    
    cc=hsv(nlines);
    for j=1:length(s)
        
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(angelsList(rightT(s(j)),1:2),1), face(angelsList(rightT(s(j)),1:2),2), face(angelsList(rightT(s(j)),1:2),3),'Color',cc(j,:),'LineWidth',3);hold on;
        plot3(face(angelsList(rightT(s(j)),2:3),1),  face(angelsList(rightT(s(j)),2:3),2), face(angelsList(rightT(s(j)),2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        %         pause
        
    end
    hold off;
elseif flag==33
    nlines=length(s);
    
    cc=hsv(nlines);
    for j=1:length(s)
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        plot3(face(angelsList(leftT(s(j)),1:2),1), face(angelsList(leftT(s(j)),1:2),2), face(angelsList(leftT(s(j)),1:2),3),'Color',cc(j,:),'LineWidth',3);hold on;
        plot3(face(angelsList(leftT(s(j)),2:3),1),  face(angelsList(leftT(s(j)),2:3),2), face(angelsList(leftT(s(j)),2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        %         pause
        
    end
    hold off;
elseif flag==4%plot tringles
    %     load KinctTri;angelsList=KinctTri;
    nlines=length(s);
    cc=hsv(nlines);
    for j=1:length(s)
        %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
        %         plot3(face(angelsList(s(j),1:2),1), face(angelsList(s(j),1:2),2), face(angelsList(s(j),1:2),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %         plot3(face(angelsList(s(j),2:3),1),  face(angelsList(s(j),2:3),2), face(angelsList(s(j),2:3),3),'Color',cc(j,:),'LineWidth',3);hold on;
        %         plot3(face(angelsList(s(j),[1,3]),1),  face(angelsList(s(j),[1 3]),2), face(angelsList(s(j),[1 3]),3),'Color',cc(j,:),'LineWidth',3);hold on;
        fill3(face(angelsList(s(j),:),1),  face(angelsList(s(j),:),2), face(angelsList(s(j),:),3),cc(j,:));hold on;
        % text(face(angelsList(s(j),1),1),  face(angelsList(s(j),1),2), face(angelsList(s(j),1),3)-0.02, num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
        %         pause
    end
    hold off;
    
elseif flag==42%plot tringles
    %     load KinctTri;angelsList=KinctTri;
    nlines=length(s);
    cc=hsv(nlines);
    for j=1:length(s)
        fill3(face(angelsList(rightT(s(j)),:),1),  face(angelsList(rightT(s(j)),:),2), face(angelsList(rightT(s(j)),:),3),cc(j,:));hold on;
    end
    hold off;
    
elseif flag==43%plot tringles
    %     load KinctTri;angelsList=KinctTri;
    nlines=length(s);
    cc=hsv(nlines);
    for j=1:length(s)
        fill3(face(angelsList(leftT(s(j)),:),1),  face(angelsList(leftT(s(j)),:),2), face(angelsList(leftT(s(j)),:),3),cc(j,:));hold on;
        
    end
    hold off;
elseif flag==5%plot all angles
    %     load KinctTri;angelsList=KinctTri;
    nlines=length(s);
    ix=mod(s,3);
    cc=hsv(3*size(angelsList,1));
    for j=1:length(s)
        if ix(j)==0
            %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
            plot3(face(angelsList(fix(s(j)/3),1:2),1), face(angelsList(fix(s(j)/3),1:2),2), face(angelsList(fix(s(j)/3),1:2),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            plot3(face(angelsList(fix(s(j)/3),2:3),1),  face(angelsList(fix(s(j)/3),2:3),2), face(angelsList(fix(s(j)/3),2:3),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
            %         pause
        elseif ix(j)==1
            %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
            plot3(face(angelsList(fix(s(j)/3),[1,3]),1), face(angelsList(fix(s(j)/3),[1,3]),2), face(angelsList(fix(s(j)/3),[1,3]),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            plot3(face(angelsList(fix(s(j)/3),2:3),1),  face(angelsList(fix(s(j)/3),2:3),2), face(angelsList(fix(s(j)/3),2:3),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
            %
        elseif ix(j)==2
            %     text(face(i,1),face(i,2),face(i,3), num2str(i-1), 'Color', 'b','FontSize',18);hold on
            plot3(face(angelsList(fix(s(j)/3),1:2),1), face(angelsList(fix(s(j)/3),1:2),2), face(angelsList(fix(s(j)/3),1:2),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            plot3(face(angelsList(fix(s(j)/3),[1,3]),1),  face(angelsList(fix(s(j)/3),[1,3]),2), face(angelsList(fix(s(j)/3),[1,3]),3),'Color',cc(s(j),:),'LineWidth',3);hold on;
            %  text(face(angelsList(s(j),1),1), face(angelsList(s(j),1),2), face(angelsList(s(j),1),3), num2str(s(j)), 'Color', cc(j,:),'FontSize',14);hold on
            %
        end
    end
    hold off;
end
view(0,260);
% title(titl)
end