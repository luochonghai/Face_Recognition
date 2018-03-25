function [ret, az, el]= validString2(str)

defaultAz = 30;
defaultEl = 30;
az = 0; el = 0;

if ischar(str)
  c1 = lower(str(1));
  
  if length(str)>1
    c2 = lower(str(2));
  else
    c2 = [];
  end
  
  if c1=='r'        %right
    ret = 1;
    az = defaultAz; el = defaultEl;
  elseif c1=='h'    %headlight
    ret = 1;
    az = 0; el = 0;
  elseif c1=='i'    %infinite
    ret = 2;
  elseif c1=='l' && ~isempty(c2)
    if c2=='o'      %local
      ret = 2;
    elseif c2=='e'  %left
      ret = 1;
      az = -defaultAz; el = defaultEl;
    else
      ret = 0;
    end
  else
    ret = 0;
  end
else
  ret = 0;
end
