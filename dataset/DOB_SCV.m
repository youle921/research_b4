function [Fold]=DOB_SCV(data,Targets,k)
ds=[data,Targets];
C=unique(Targets);
Fold={};
f=1;
for i=1:length(C)
    ind=find(ds(:,end)==C(i));
    ind_mask=ones(length(ind),1);
   
    while sum(ind_mask)~=0
        ind=ind.*ind_mask;
        ind=nonzeros(ind);
        ind_mask=nonzeros(ind_mask);
        L=length(find(ind_mask==1));
        E_0=randi(L);
        E_0=repmat(data(ind(E_0),:),[L,1]);
        dist=sqrt(sum(((data(ind,:)-E_0).^2)'));
        [D,I]=sort(dist);
        if length(I)<k
            k_new=length(I);
        else
            k_new=k;
        end
        for j=1:k_new
            if f==1
                Fold{j,1}=[ind(I(j))];
            else
                Fold{j}=[Fold{j},ind(I(j))];
            end
        end
        f=0;
        [~,ind_Remove]=ismember(ind(I(1:k_new)),ind);
        ind_mask(ind_Remove)=0;
    end
end

end