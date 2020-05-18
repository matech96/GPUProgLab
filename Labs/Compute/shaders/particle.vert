#version 430

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec4 vVelocity;

out vec4 fVelocity;
out vec4 pos;

void main()
{
	gl_Position = vPosition;
	pos = vPosition;
	fVelocity = vVelocity;
}