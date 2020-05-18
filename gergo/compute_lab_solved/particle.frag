#version 430

in vec4 fVelocity;

in vec4 pos;
out vec4 outColor;
uniform vec2 click;

void main()
{
	if(length(fVelocity) > 0.0) outColor = vec4(0,1,0,1);
	else outColor = vec4(0,0,0,1);

	//outColor = vec4((fVelocity.xyz + 1.0) / 2.0, 1.0);
}