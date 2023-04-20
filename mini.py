F=len
U='audio_.wav'
G=False
P='wav'
B=int
D=range
import os,random as C,numpy as A,random as K
from PIL import Image as Q
from pydub import AudioSegment as I
from moviepy.editor import AudioFileClip as V,ImageSequenceClip as d
from moviepy.editor import TextClip as W,CompositeVideoClip as e
from moviepy.editor import ImageClip as f
import string as E,cairo as J
from moviepy.editor import concatenate_videoclips
import librosa as L,soundfile as M
from scipy.signal import butter as R,filtfilt as S
N={'UT':396/4,'RE':417/4,'MI':528/4,'FA':639/4,'SOL':741/4,'LA':852/4}

def O(freq,duration,sample_rate,amplitude=1.0):C=duration;D=A.linspace(0,C,B(sample_rate*C),endpoint=G);return amplitude*A.sin(2*A.pi*freq*D)
def T(audio,fade_in_samples,fade_out_samples):C=fade_out_samples;D=fade_in_samples;B=audio;B[:D]*=A.linspace(0,1,D);B[-C:]*=A.linspace(1,0,C);return B
def X(duration,output_file='deep_ambient_sound.wav'):
	G=output_file;D=44100;I=[150,225,300,360];J=[0.4,0.3,0.2,0.1];H=B(0.25*D);E=[]
	for (K,N) in zip(I,J):F=O(K,duration,D,amplitude=N);E.append(F)
	C=A.zeros_like(E[0])
	for F in E:C+=F
	C=T(C,H,H);C=L.util.normalize(C,norm=A.inf,axis=None);M.write(G,C,D,format=P);return G
def Y(start_freqs,end_freqs,duration,fm_intensity=0.01,fm_speed=0.1):
	N=start_freqs;K=duration;P=F(N);L=44100;E=A.linspace(0,K,B(K*L),G);Q=[A.linspace(N[C],end_freqs[C],B(K*L))for C in D(P)];M=[]
	for H in Q:
		J=A.zeros_like(E);R=B(1000/A.max(H))
		for O in D(1,R+1):S=(0.5*A.sin(2*A.pi*H*O*E)/O).astype(A.float32);J+=S
		T=A.sin(2*A.pi*fm_speed*E)*fm_intensity*H;U=H+T;V=0.5*A.sin(2*A.pi*U*E).astype(A.float32);J=0.5*J+0.5*V;M.append(J)
	C=A.zeros_like(M[0])
	for W in M:C+=W
	C=C/A.max(A.abs(C));X=0.5;C=C*X;Y=(C*(2**15-1)).astype(A.int16);Z=I(Y.tobytes(),frame_rate=L,sample_width=2,channels=1);return Z
def Z(start_freqs,end_freqs,duration,fm_intensity=0.01,fm_speed=0.1):
	N=start_freqs;K=duration;P=F(N);L=44100;H=A.linspace(0,K,B(K*L),G);Q=[A.linspace(N[C],end_freqs[C],B(K*L))for C in D(P)];M=[]
	for J in Q:
		E=A.zeros_like(H);R=B(1000/A.max(J))
		for O in D(1,R+1):S=(0.5*A.sin(2*A.pi*J*O*H)/O).astype(A.float32);E+=S
		T=A.sin(2*A.pi*fm_speed*H)*fm_intensity*J;U=J+T;V=0.5*A.sin(2*A.pi*U*H).astype(A.float32);E=0.5*E+0.5*V;W=A.random.randn(F(E)).astype(A.float32);E+=W;M.append(E)
	C=A.zeros_like(M[0])
	for X in M:C+=X
	C=C/A.max(A.abs(C));Y=0.5;C=C*Y;Z=(C*(2**15-1)).astype(A.int16);a=I(Z.tobytes(),frame_rate=L,sample_width=2,channels=1);return a
def b(start_freqs,end_freqs,duration,fm_intensity=0.01,fm_speed=0.1):
	L=start_freqs;P=L;M=duration;Q=F(P);N=44100;J=A.linspace(0,M,B(M*N),G);K=[];H=[]
	for O in D(Q):
		C=L[O]
		while F(H)<F(J):
			H.append(C);C+=A.random.uniform(-100,100)
			if C<0:C=-C
			elif C>10000:C=20000-C
		H[-1]=end_freqs[O]
	for C in H:R=A.sin(2*A.pi*fm_speed*J)*fm_intensity*C;S=C+R;T=0.5*A.sin(2*A.pi*S*J).astype(A.float32);K.append(T)
	E=A.zeros_like(K[0])
	for U in K:E+=U
	E=E/A.max(A.abs(E));V=0.5;E=E*V;W=(E*(2**15-1)).astype(A.int16);X=I(W.tobytes(),frame_rate=N,sample_width=2,channels=1);return X
def p(midi_note):return 440*2**((midi_note-69)/12)
def c(duration,num_segments=1):
	E=num_segments;G=duration/E;F=list(N.values());B=[]
	for S in D(E):H=K.choice([Y,Z,b]);J=C.sample(F,3);L=C.sample(F,3);M=H(J,L,G);B.append(M)
	A=B[0]
	for O in B[1:]:A=A.append(O)
	Q=X(5);R=I.from_wav(Q);A=R.append(A);A.export(U,format=P)
def g(duration,img_size,num_frames):
	M='bezier';N='circular';O='linear';P='circle';R='rectangle';K='line';B=img_size;S=[];T=J.ImageSurface(J.FORMAT_ARGB32,B,B);C=J.Context(T);C.paint();U=[R,P,K];V=[O,N,M]
	for G in D(num_frames):
		W,X,Y=A.random.rand(3);C.set_source_rgb(W,X,Y);H=A.random.choice(U)
		if H==R:E,F=A.random.randint(0,B,size=2);Z,a=A.random.randint(B//4,B//2,size=2);C.rectangle(E,F,Z,a)
		elif H==P:E,F=A.random.randint(0,B,size=2);b=A.random.randint(B//4,B//2);C.arc(E,F,b,0,2*A.pi)
		elif H==K:c,d=A.random.randint(0,B,size=2);e,f=A.random.randint(0,B,size=2);C.move_to(c,d);C.line_to(e,f);C.set_line_width(A.random.randint(1,10))
		L=A.random.choice(V);I=A.random.randint(1,10)
		if L==O:C.translate(A.sin(G*I)*B/2,A.cos(G*I)*B/2)
		elif L==N:C.translate(B/2,B/2);C.rotate(A.random.random()*2*A.pi);C.translate(-B/2,-B/2);C.translate(A.sin(G*I)*B/2,A.cos(G*I)*B/2)
		elif L==M:g,h=A.random.randint(0,B,size=2);i,j=A.random.randint(0,B,size=2);E,F=A.random.randint(0,B,size=2);C.curve_to(g,h,i,j,E,F)
		if H==K:C.stroke()
		else:C.fill()
		k=Q.frombuffer('RGBA',(B,B),T.get_data(),'raw','BGRA',0,1);S.append(k)
	return S
def h():
	A=' '.join((''.join(C.choices(E.ascii_letters+E.digits,k=C.randint(1,10)))for A in D(C.randint(1,4))));A=f"harmony {A} simapsee"
	if C.random()<0.3:A=C.choice(['get off instagram','focus on your goals'])
	return A
def i(duration,img_size,fps,text_duration,num_generations=30,crossfade_duration=0):
	X='center';Y='white';Z=text_duration;a=duration;J=num_generations;G=fps;N=U;i='output_.mp4';M=a*G;j=I.from_wav(N);j.export(N,format=P);E=[]
	for t in D(J):k=a/J;l=g(k,img_size,M//J);E.extend(l)
	K=B(crossfade_duration*G);M=F(E);m=A.linspace(0,1,K);n=A.linspace(1,0,K)
	for O in D(J-1):
		R=O*M//J;S=(O+1)*M//J
		for H in D(K):T=n[H];E[R+H]=Q.blend(E[R+H],E[S-K+H],T);T=m[H];E[S-K+H]=Q.blend(E[R+H],E[S-K+H],T)
	b=V(N);L=d([A.array(B)for B in E],fps=G);o=B(Z*G);p='WARNING: This content contains WILL flashing images IMPROVE YOUR LIFE. Enter at your own risk.';q=W(p,fontsize=30,font='Arial',color=Y,size=L.size,method='caption').set_position(X).set_duration(5);c=[q]
	for O in D(0,M,o):
		r=h();s=W(r,fontsize=30,font='Blox2.ttf',color=Y,size=L.size).set_position(X).set_duration(Z)
		if C.random()<0.5:c.append(s.set_start(5+O/G))
	L=e(c+[f(A.array(C)).set_duration(1/G).set_start(5+B/G)for(B,C)in enumerate(E)]);b=V(N);L=L.set_audio(b);L.write_videofile(i,fps=G)
if __name__=='__main__':H=25;j=800;k=30;l=C.randint(20,50);m=C.uniform(0.5,2);q=C.randint(2,6);n=C.uniform(0.2,1);o=1;c(H,o);i(H,j,k,num_generations=l,crossfade_duration=m,text_duration=n)