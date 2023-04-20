Z='audio_.wav'
Y='center'
K=False
Q='wav'
F=int
E=range
import os,random as B,numpy as A
from PIL import Image as P
from pydub import AudioSegment as O
from moviepy.editor import AudioFileClip as c,ImageSequenceClip as d
from moviepy.editor import TextClip as W,CompositeVideoClip as e
import string as X,cairo as L
from moviepy.editor import concatenate_videoclips
import librosa as I,soundfile as J
G={'UT':396/4,'RE':417/4,'MI':528/4,'FA':639/4,'SOL':741/4,'LA':852/4}
def M(freq,duration,sample_rate,amplitude=1.0):B=duration;C=A.linspace(0,B,F(sample_rate*B),endpoint=K);return amplitude*A.sin(2*A.pi*freq*C)
def N(audio,fade_in_samples,fade_out_samples):D=fade_out_samples;C=fade_in_samples;B=audio;B[:C]*=A.linspace(0,1,C);B[-D:]*=A.linspace(1,0,D);return B
def H(duration,output_file='deep_ambient_sound.wav'):
	G=output_file;C=44100;K=[150,225,300,360];L=[0.4,0.3,0.2,0.1];H=F(0.25*C);D=[]
	for (O,P) in zip(K,L):E=M(O,duration,C,amplitude=P);D.append(E)
	B=A.zeros_like(D[0])
	for E in D:B+=E
	B=N(B,H,H);B=I.util.normalize(B,norm=A.inf,axis=None);J.write(G,B,C,format=Q);return G
def f(duration,text,fontsize,font,color,size,position=Y):A=W(text,fontsize=fontsize,font=font,color=color,size=size).set_position(position).set_duration(duration);return A
def R(start_freqs,end_freqs,duration,fm_intensity=0.01,fm_speed=0.1):
	L=start_freqs;H=duration;N=len(L);I=44100;C=A.linspace(0,H,F(H*I),K);P=[A.linspace(L[B],end_freqs[B],F(H*I))for B in E(N)];J=[]
	for D in P:
		G=A.zeros_like(C);Q=F(1000/A.max(D))
		for M in E(1,Q+1):R=(0.5*A.sin(2*A.pi*D*M*C)/M).astype(A.float32);G+=R
		S=A.sin(2*A.pi*fm_speed*C)*fm_intensity*D;T=D+S;U=0.5*A.sin(2*A.pi*T*C).astype(A.float32);G=0.5*G+0.5*U;J.append(G)
	B=A.zeros_like(J[0])
	for V in J:B+=V
	B=B/A.max(A.abs(B));W=0.5;B=B*W;X=(B*(2**15-1)).astype(A.int16);Y=O(X.tobytes(),frame_rate=I,sample_width=2,channels=1);return Y
def j(midi_note):return 440*2**((midi_note-69)/12)
def D(duration,num_segments=1):
	D=num_segments;I=duration/D;F=list(G.values());C=[]
	for S in E(D):J=B.sample(F,3);K=B.sample(F,3);L=R(J,K,I);C.append(L)
	A=C[0]
	for M in C[1:]:A=A.append(M)
	N=H(5);P=O.from_wav(N);A=P.append(A);A.export(Z,format=Q)
def g(duration,img_size,num_frames,update_interval=5):
	o='color';n='scale';m='bezier';l='circular';k='linear';j='arc';i='triangle';h='circle';g='rectangle';W='line';Q=update_interval;I='params';H='type';C=img_size;X=[];Y=L.ImageSurface(L.FORMAT_ARGB32,C,C);B=L.Context(Y);p=[g,h,W,i,j];Z=[k,l,m,n];a=[];M=[]
	for J in E(num_frames):
		B.save();B.set_operator(L.Operator.CLEAR);B.paint();B.restore()
		if J%Q==0:
			a=M;M=[];q=A.random.randint(1,5)
			for r in E(q):D={H:A.random.choice(p),I:A.random.uniform(0,C,size=4),o:A.random.uniform(0.3,0.7,size=4),'movement':A.random.choice(Z),'speed':A.random.randint(1,10)};M.append(D)
		b=J%Q/Q;c=[]
		for (d,e) in zip(a,M):
			D={}
			for K in d.keys():
				if K==I:D[K]=d[K]*(1-b)+e[K]*b
				else:D[K]=e[K]
			c.append(D)
		for D in c:
			B.set_source_rgba(*D[o])
			if D[H]==g:F,G,s,t=D[I];B.rectangle(F,G,s,t)
			elif D[H]==h:F,G,R,r=D[I];B.arc(F,G,R,0,2*A.pi)
			elif D[H]==W:S,T,U,V=D[I];B.move_to(S,T);B.line_to(U,V);B.set_line_width(A.random.randint(1,10))
			elif D[H]==i:S,T,U,V=D[I];u,v=A.random.randint(0,C,size=2);B.move_to(S,T);B.line_to(U,V);B.line_to(u,v);B.close_path()
			elif D[H]==j:F,G,R,w=D[I];B.arc(F,G,R,0,w)
			N=A.random.choice(Z);O=A.random.randint(1,10)
			if N==k:B.translate(A.sin(J*O)*C/2,A.cos(J*O)*C/2)
			elif N==l:B.translate(C/2,C/2);B.rotate(A.random.random()*2*A.pi);B.translate(-C/2,-C/2);B.translate(A.sin(J*O)*C/2,A.cos(J*O)*C/2)
			elif N==m:x,y=A.random.randint(0,C,size=2);z,A0=A.random.randint(0,C,size=2);F,G=A.random.randint(0,C,size=2);B.curve_to(x,y,z,A0,F,G)
			elif N==n:f=A.random.rand()*2;B.translate(C/2,C/2);B.scale(f,f);B.translate(-C/2,-C/2)
			if D[H]==W:B.stroke()
			else:B.fill()
		A1=P.frombuffer('RGBA',(C,C),Y.get_data(),'raw','BGRA',0,1);X.append(A1)
	return X
def S(duration,img_size,fps,text_interval,text_duration,num_generations=30,crossfade_duration=0):
	b='white';a=duration;R=text_duration;H=num_generations;G=fps;S=Z;h='output_.mp4';K=a*G;i=O.from_wav(S);i.export(S,format=Q);C=[]
	for s in E(H):j=a/H;k=g(j,img_size,K//H);C.extend(k)
	I=F(crossfade_duration*G);K=len(C);l=A.linspace(0,1,I);m=A.linspace(1,0,I)
	for L in E(H-1):
		T=L*K//H;U=(L+1)*K//H
		for D in E(I):V=m[D];C[T+D]=P.blend(C[T+D],C[U-I+D],V);V=l[D];C[U-I+D]=P.blend(C[T+D],C[U-I+D],V)
	n=c(S);J=d([A.array(B)for B in C],fps=G);R=0.5;o=F(R*G);p='WARNING: This content will improve your life. Enter at your own risk.';q=f(5,p,30,'Arial',b,J.size);M=[];M=[q]+M
	for L in E(0,K,o):
		N=' '.join((''.join(B.choices(X.ascii_letters+X.digits,k=B.randint(1,10)))for A in E(B.randint(1,4))));N=f"harmony {N} simapsee"
		if B.random()<0.3:N=B.choice(['get off instagram','focus on your goals'])
		r=W(N,fontsize=30,font='Blox2.ttf',color=b,size=J.size).set_position(Y).set_duration(R);M.append(r.set_start(L/G))
	J=e(M);J=J.set_audio(n);J.write_videofile(h,fps=G)
if __name__=='__main__':C=30;T=800;U=30;V=B.randint(20,50);a=B.uniform(0.5,2);b=B.randint(2,6);h=B.uniform(0.2,1);i=1;D(C,i);S(C,T,U,num_generations=V,crossfade_duration=a,text_duration=h,text_interval=b)