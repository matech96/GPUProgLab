#version 330

uniform sampler2D data;
uniform vec4 ourColor;

//in vec2 fTexCoord;
out vec4 outColor;

// Fraktal
/*void main()
{
	// outColor = texture(data, fTexCoord);
	int n = 20;
	float max = 1;
	vec2 z = vec2(0,0);
	for(int i=0;i<n;i++){
		z = vec2( (z.x*z.x) - (z.y*z.y) + (fTexCoord.x*4-2),
		 (2*z.x*z.y) + (fTexCoord.y*4-2));
	}
	if (sqrt((z.x*z.x) + (z.y*z.y)) > max){
		outColor = vec4(1,1,1,0);
	} else {
		outColor = vec4(0,0,0,0);
	}
	// outColor = vec4(fTexCoord.xy, 1, 1);
}*/

// gray scale
/*void main(){
	float tmp = dot(texture(data, fTexCoord), vec4(0.21, 0.39, 0.4, 0));
	outColor = vec4(tmp, tmp, tmp, 0);
}*/

float getGray(vec2 coord){
	return dot(texture(data, coord), vec4(0.21, 0.39, 0.4, 0));
}

// basic edge
//void main(){
//	float d = 0.001;
//	float L_dx = (getGray(fTexCoord + vec2(d,0)) - getGray(fTexCoord - vec2(d,0))) / 2.0;
//	float L_dy = (getGray(fTexCoord + vec2(0,d)) - getGray(fTexCoord - vec2(0,d))) / 2.0;
//	//outColor = vec4(L_dx+L_dy, L_dx+L_dy, L_dx+L_dy, 0);
//	float tmp = sqrt((L_dx*L_dx)+(L_dy*L_dy)); // 0.5;
//	outColor = vec4(tmp,tmp,tmp, 0);
//}

//laplace
//void main(){
//	float d = 0.01;
//	float tmp = (getGray(fTexCoord + vec2(d,0))
//					+ getGray(fTexCoord - vec2(d,0))
//					+ getGray(fTexCoord + vec2(0,d))
//					+ getGray(fTexCoord - vec2(0,d)))
//						- (4*getGray(fTexCoord));
//	//outColor = vec4(L_dx+L_dy, L_dx+L_dy, L_dx+L_dy, 0);
////	float tmp = sqrt((L_dx*L_dx)+(L_dy*L_dy)); // 0.5;
//	outColor = vec4(tmp,tmp,tmp, 0);
//}

//elkiemeles
void main(){
//	float d = 2.0 / 600.0; //0.005;
//	outColor = texture(data, fTexCoord)
//				- (texture(data, fTexCoord + vec2(d,0))
//					+ texture(data, fTexCoord - vec2(d,0))
//					+ texture(data, fTexCoord + vec2(0,d))
//					+ texture(data, fTexCoord - vec2(0,d)))
//						+ (4*texture(data, fTexCoord));
	outColor = ourColor;
	//outColor = vec4(L_dx+L_dy, L_dx+L_dy, L_dx+L_dy, 0);
//	float tmp = sqrt((L_dx*L_dx)+(L_dy*L_dy)); // 0.5;
//	outColor = vec4(tmp,tmp,tmp, 0);
}