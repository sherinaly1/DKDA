function DS=composeDS(X,Y,subjects,name,sequence)
% X: (N x dim ) 
% DS.input: (dim x N)

DS.input=X';DS.output=Y';DS.subjects=subjects';DS.name=name;
if nargin==5
DS.sequence=sequence';
end
end