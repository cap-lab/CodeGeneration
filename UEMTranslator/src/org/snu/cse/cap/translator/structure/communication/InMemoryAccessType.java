package org.snu.cse.cap.translator.structure.communication;

public enum InMemoryAccessType {
	CPU_ONLY,
	GPU_ONLY,
	CPU_GPU,
	GPU_CPU,
	GPU_GPU,
	GPU_GPU_DIFFERENT,
}
