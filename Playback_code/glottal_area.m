function [vGLA]=glottal_area(vmAreas)
%[vGLA]=glottal_area(vmAreas)
% Calcula el �rea de la glotis de cada imagen y lo devuelve en un vector
%
% INPUTS:
%       - vmAreas: vector de matrices que incluye todas las im�genes de la
%       glotis. El bakground est� a 0 y el �rea de la glotis, a 1.
%
% OUTPUTS: 
%       - vGLA: vector que contiene el �rea de la glotis en cada instante
%
% $Id:$

[~, ~,iNumImages]=size(vmAreas);
vGLA=zeros(1,iNumImages);
for i=1:iNumImages
    vGLA(i)=sum(sum(vmAreas(:,:,i)));
end
end
    
    
    