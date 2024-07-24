function[Trajectory]=compute_trajectory(edges,Dorsal,Ventral,Porc_GAxis,sizeVideo)

      
 for i=1:sizeVideo.nrframes
    D_Vx(i)=abs(Dorsal(i,2)-Ventral(i,2));%2 is X axis and 1 is Y axis-- 
    rectP.point(1,:)=0:10:sizeVideo.col;%line perpendicular to the glottal axis(x coordinates)
    L(i)=sqrt((Dorsal(i,1)-Ventral(i,1)).^2+(Dorsal(i,2)-Ventral(i,2)).^2);% glottal axis longuitud
    g_100(i)=Dorsal(i,1)+L(i);
    Angle_rot=asin(D_Vx(i)/L(i));% angle reference for transform coordinates
    m_g=((Dorsal(i,1)-Ventral(i,1))/(Dorsal(i,2)-Ventral(i,2)));%pendiente glottal axis
    
    g.y(i)=Dorsal(i,1)+((L(i)).*(Porc_GAxis/100)).*cos(Angle_rot);%Y coordinate over glottal axis
    if m_g<0
    g.x(i)=Dorsal(i,2)-(tan(Angle_rot)).*(g.y(i)-Dorsal(i,1));%X coordinate over glottal axis
    else
    g.x(i)=Dorsal(i,2)+(tan(Angle_rot)).*(g.y(i)-Dorsal(i,1));
    end
    
    rectP.m=-1/m_g; %slope of line perpendicular to glottal axis 
    rectP.b=g.y(i)-rectP.m*(g.x(i)); %b of the line perpendicular to glottal axis
    rectP.point(2,:)=(rectP.m).*rectP.point(1,:)+rectP.b;%ecuacion de la recta puntos en y
    
    VF_right(2,:)=edges(i).left(:,1);%the left vocal edge is the right for me and viceversa
    VF_right(1,:)=edges(i).left(:,2);
    VF_left(2,:)=edges(i).right(:,1);
    VF_left(1,:)=edges(i).right(:,2);
    VFL_in = InterX(rectP.point,VF_left);%intersection between 2 rect in this case the perpendicular to the axis and the VFs
    VFR_in= InterX(rectP.point,VF_right);
    
    Trajectory.VFL(i)=sqrt((VFL_in(1,1)-g.x(i)).^2+(VFL_in(2,1)-g.y(i)).^2);
    Trajectory.VFR(i)=sqrt((VFR_in(1,1)-g.x(i)).^2+(VFR_in(2,1)-g.y(i)).^2);
    Trajectory.VFL_in(i,:)=VFL_in(:,1);
    Trajectory.VFR_in(i,:)=VFR_in(:,1);
    Trajectory.g(i,1)=g.x(i);
    Trajectory.g(i,2)=g.y(i);
    Trajectory.LineP(i)=rectP;
    
        if VFL_in(1,1)>g.x(i)
            Trajectory.VFL(i)=-Trajectory.VFL(i);
        end
        if VFR_in(1,1)>g.x(i)
            Trajectory.VFR(i)=-Trajectory.VFR(i);
        end
    
        
     figure(1);
     plot([Dorsal(i,2),Ventral(i,2)],[Dorsal(i,1),Ventral(i,1)]); hold on;
     plot([Dorsal(i,2),Dorsal(i,2)],[Dorsal(i,1),g_100(i)]);hold on;
     plot(g.x(i),g.y(i),'*');hold on;
     plot(Dorsal(i,2),g.y(i),'*');hold on;
     plot(rectP.point(1,:),rectP.point(2,:));hold on;
     plot(edges(i).left(:,2),edges(i).left(:,1),'r');hold on;%plot right edges right
     plot(edges(i).right(:,2),edges(i).right(:,1),'b');hold on;%plot left edges blue
     plot(VFL_in(1,1),VFL_in(2,1),'o','MarkerSize',10,'MarkerEdgeColor','b');hold on;
     plot(VFR_in(1,1),VFR_in(2,1),'o','MarkerSize',10,'MarkerEdgeColor','r');hold off;
     title(['TRAJECTORY - Frame:' num2str(i)]);
     ylim([0 sizeVideo.row])
     set(gca,'YDir','reverse');
     Trajectory.video(i) = getframe(figure(1));
        
 end
    

