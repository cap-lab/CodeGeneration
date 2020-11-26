package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.BoundType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICProfileTypeLoader;
import hopes.cic.xml.ProfileCommType;
import hopes.cic.xml.ProfileExecutionBoundType;
import hopes.cic.xml.ProfileLibraryFunctionType;
import hopes.cic.xml.ProfileLibraryType;
import hopes.cic.xml.ProfileMigrationType;
import hopes.cic.xml.ProfileTaskModeType;
import hopes.cic.xml.ProfileTaskType;
import hopes.cic.xml.ProfileType;

public class CICProfileXMLHandler extends CICXMLHandler{
	private CICProfileTypeLoader loader;
	private CICProfileType profile;

	private List<ProfileTaskType> taskList = new ArrayList<ProfileTaskType>();
	private List<ProfileLibraryType> libraryList = new ArrayList<ProfileLibraryType>();
	private List<String> procList = new ArrayList<String>();
	private List<ProfileCommType> commList = new ArrayList<ProfileCommType>();
	private List<ProfileMigrationType> migrationList = new ArrayList<ProfileMigrationType>();

	public CICProfileXMLHandler() {
		loader = new CICProfileTypeLoader();
		profile = new CICProfileType();
	}

	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(profile, writer);
	}

	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		profile = loader.loadResource(is);
	}

	public void init() {
		makeTaskList();
		makeLibraryList();
		makeProcList();
		makeCommList();
		makeMigrationList();
	}

	public CICProfileType getProfile() {
		return profile;
	}

	public List<ProfileTaskType> getTaskList() {
		return taskList;
	}

	public void setTaskList(List<ProfileTaskType> taskList) {
		this.taskList = taskList;
	}

	public List<ProfileLibraryType> getLibraryList() {
		return libraryList;
	}

	public void setLibraryList(List<ProfileLibraryType> libraryList) {
		this.libraryList = libraryList;
	}

	public List<ProfileCommType> getCommList() {
		return commList;
	}

	public void setCommList(List<ProfileCommType> commList) {
		this.commList = commList;
	}

	public List<ProfileMigrationType> getMigrationList() {
		return migrationList;
	}

	public void setMigrationList(List<ProfileMigrationType> migrationList) {
		this.migrationList = migrationList;
	}

	public void update_profile() {
		profile.getTask().clear();
		profile.getComm().clear();
		taskList.forEach(t -> profile.getTask().add(t));
		commList.forEach(c -> profile.getComm().add(c));
	}

	public void makeTaskList() {
		taskList.clear();
		for (ProfileTaskType taskType : profile.getTask()) {
			String taskName = taskType.getName();
			ProfileTaskType task = new ProfileTaskType();
			task.setName(taskName);

			for (ProfileTaskModeType taskModeType : taskType.getMode()) {
				ProfileTaskModeType taskMode = new ProfileTaskModeType();
				taskMode.setName(taskModeType.getName());
				for (ProfileType profileType : taskModeType.getProfile()) {
					taskMode.getProfile().add(profileType);
				}
				task.getMode().add(taskMode);
			}
			taskList.add(task);
		}
	}

	public void makeLibraryList() {
		libraryList.clear();
		for (ProfileLibraryType libraryType : profile.getLibrary()) {
			String libraryName = libraryType.getName();
			ProfileLibraryType library = new ProfileLibraryType();
			library.setName(libraryName);
			for (ProfileLibraryFunctionType libraryFunctionType : libraryType.getFunction()) {
				ProfileLibraryFunctionType libFunc = new ProfileLibraryFunctionType();
				libFunc.setName(libraryFunctionType.getName());
				for (ProfileType profileType : libraryFunctionType.getProfile()) {
					libFunc.getProfile().add(profileType);
				}
				library.getFunction().add(libFunc);
			}
			libraryList.add(library);
		}
	}

	public void makeProcList() {
		if (profile.getTask().isEmpty()) {
			return;
		}
		ProfileTaskType task = profile.getTask().get(0);
		for (ProfileType pt : task.getMode().get(0).getProfile()) {
			procList.add(pt.getProcessorType());
		}
	}

	public void makeCommList() {
		commList.clear();
		for (ProfileCommType commType : profile.getComm()) {
			String srcProc = commType.getSrc();
			String dstProc = commType.getDst();
			ProfileCommType comm = new ProfileCommType();
			comm.setSrc(srcProc);
			comm.setDst(dstProc);
			comm.setSecondPowerCoef(commType.getSecondPowerCoef());
			comm.setFirstPowerCoef(commType.getFirstPowerCoef());
			comm.setConstant(commType.getConstant());
			comm.setTimeunit(commType.getTimeunit());
			commList.add(comm);
		}
	}

	public void makeMigrationList() {
		migrationList.clear();
		for (ProfileMigrationType migrationType : profile.getMigration()) {
			String srcProc = migrationType.getSrc();
			String dstProc = migrationType.getDst();

			ProfileMigrationType migration = new ProfileMigrationType();

			migration.setSrc(srcProc);
			migration.setDst(dstProc);
			migration.setCost(migrationType.getCost());
			migration.setSettingcost(migrationType.getSettingcost());
			migration.setTimeunit(migrationType.getTimeunit());
			migration.setSize(migrationType.getSize());
			migration.setSizeunit(migrationType.getSizeunit());
			migrationList.add(migration);
		}
	}

	public ProfileTaskType findTaskByName(String taskName) {
		for (ProfileTaskType task : taskList) {
			if (task.getName().equals(taskName)) {
				return task;
			}
		}
		return null;
	}

	public ProfileLibraryType findLibraryByName(String libraryName) {
		for (ProfileLibraryType library : libraryList) {
			if (library.getName().equals(libraryName)) {
				return library;
			}
		}
		return null;
	}

	public ProfileCommType findCommByName(String srcProc, String dstProc) {
		for (ProfileCommType comm : commList) {
			if (comm.getSrc().equals(srcProc) && comm.getDst().equals(dstProc)) {
				return comm;
			}
		}
		return null;
	}

	public ProfileMigrationType findMigrationByName(String srcProc, String dstProc) {
		for (ProfileMigrationType migration : migrationList) {
			if (migration.getSrc().equals(srcProc) && migration.getDst().equals(dstProc)) {
				return migration;
			}
		}
		return null;
	}

	public ProfileExecutionBoundType getOrMakeExecutionBound(ProfileType pt, BoundType bound) {
		Optional<ProfileExecutionBoundType> boundOptional = pt.getBound().stream()
				.filter(b -> b.getType().equals(bound)).findFirst();
		if (boundOptional.isPresent()) {
			return boundOptional.get();
		} else {
			ProfileExecutionBoundType execBound = new ProfileExecutionBoundType();
			execBound.setType(bound);
			execBound.setValue(pt.getValue());
			execBound.setUnit(pt.getUnit());
			pt.getBound().add(execBound);
			return execBound;
		}
	}

	public ProfileTaskModeType getProfileTaskModeTypeByModeName(ProfileTaskType profileTask, String modeName) {
		return profileTask.getMode().stream().filter(pmt -> pmt.getName().equalsIgnoreCase(modeName)).findFirst()
				.orElseThrow(IllegalArgumentException::new);
	}

}
