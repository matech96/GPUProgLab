#version 330 core
layout (lines_adjacency) in;
layout (line_strip, max_vertices = 200) out;

void main() {	
	float s = 0.5;
	mat4 m = mat4(-s, 2.0-s, s-2.0, s,			// 1. row
                  2.0*s, s-3.0, 3.0-(2.0*s), -s,// 2. row
                  -s, 0.0, s, 0.0,				// 3. row
                  0.0, 1.0, 0.0, 0.0);			// 4. row
	m = transpose(m);
	mat4 r = mat4(gl_in[0].gl_Position,
				  gl_in[1].gl_Position,
				  gl_in[2].gl_Position,
				  gl_in[3].gl_Position);
	r = transpose(r);
	mat4 mr = m*r;
	for(int i = 0; i<200; i++){
		float t_ = float(i) / 200;
		vec4 t = vec4(t_*t_*t_, t_*t_, t_, 1.0);
		gl_Position = t*m*r;
		EmitVertex(); 		
	}

    
    EndPrimitive();
} 