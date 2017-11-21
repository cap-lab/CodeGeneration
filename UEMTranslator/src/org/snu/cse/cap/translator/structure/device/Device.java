package org.snu.cse.cap.translator.structure.device;

import java.util.ArrayList;

enum ArchitectureType {
	X86,
	X86_64,
	ARM,
	ARM64,
	GENERIC,
}

enum SoftwarePlatformType {
	ARDUINO,
	WINDOWS,
	LINUX,
	UCOS3,
}

enum RuntimeType {
	NATIVE,
	SOPHY,
	HSIM,
}

public class Device {
	private ArrayList<Processor> processorList;
	private ArrayList<Connection> connectionList;
	private ArchitectureType architecture;
	private SoftwarePlatformType platform;
	private RuntimeType runtime;

	public ArchitectureType getArchitecture() {
		return architecture;
	}
	
	public SoftwarePlatformType getPlatform() {
		return platform;
	}
	
	public RuntimeType getRuntime() {
		return runtime;
	}
	
	public void setArchitecture(ArchitectureType architecture) {
		this.architecture = architecture;
	}
	
	public void setPlatform(SoftwarePlatformType platform) {
		this.platform = platform;
	}
	
	public void setRuntime(RuntimeType runtime) {
		this.runtime = runtime;
	}
}
