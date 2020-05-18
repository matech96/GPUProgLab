#version 430

in vec4 fVelocity;

in vec4 pos;
out vec4 outColor;
uniform vec2 click;

void main()
{
	outColor = vec4(1,0,0.5,1);
}