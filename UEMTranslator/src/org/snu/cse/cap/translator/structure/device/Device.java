package org.snu.cse.cap.translator.structure.device;

import java.util.ArrayList;

enum ArchitectureType {
	X86("x86"),
	X86_64("x86_64"),
	ARM("arm"),
	ARM64("arm64"),
	GENERIC("generic"),
	;
	
	private final String value;
	
	private ArchitectureType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

enum SoftwarePlatformType {
	ARDUINO("arduino"),
	WINDOWS("windows"),
	LINUX("linux"),
	UCOS3("ucos-3"),
	;
	
	private final String value;
	
	private SoftwarePlatformType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

enum RuntimeType {
	NATIVE("native"),
	SOPHY("sophy"),
	HSIM("hsim"),
	;

	private final String value;
	
	private RuntimeType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
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
