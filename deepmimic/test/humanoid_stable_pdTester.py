import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from _pybullet_utils import pd_controller_stable
from _pybullet_env.humanoid_pose_interpolator import HumanoidPoseInterpolator
import math
import numpy as np

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
jointFrictionForce = 0


class HumanoidStablePDTester(object):
    def __init__( self, pybullet_client, 
                 mocap_data, 
                 timeStep, 
                 useFixedBase=True, 
                 useComReward=False):
        self._pybullet_client = pybullet_client
        self._mocap_data = mocap_data
        flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER+self._pybullet_client.URDF_USE_SELF_COLLISION+self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._sim_model = self._pybullet_client.loadURDF(
            "humanoid/humanoid.urdf", [0, 0.889540259, 0],
            globalScaling=0.25,
            useFixedBase=useFixedBase,
            flags=flags)
        self._end_effectors = [5, 8, 11, 14]  #ankle and wrist, both left and right
        self._pybullet_client.changeDynamics(self._sim_model, -1, lateralFriction=0.9) #linkIndex == -1: for the base
        for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
            self._pybullet_client.changeDynamics(self._sim_model, j, lateralFriction=0.9)
        self._pybullet_client.changeDynamics(self._sim_model, -1, linearDamping=0, angularDamping=0) #<??tag>

        self._poseInterpolator = HumanoidPoseInterpolator()
        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
        self._timeStep = timeStep
        self._kpOrg = [
        0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500,
        400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
        400, 400, 400, 400, 300
        ]
        print("length of self._kpOrg (Don't know what it is yet..) : ", len(self._kpOrg)) #length 43
        self._kdOrg = [
            0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40,
            40, 40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
        ]
        print("length of self._kdOrg (Don't know what it is yet..) : ", len(self._kdOrg)) #length 43
        self._jointIndicesAll = [
            chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, leftHip, leftKnee,
            leftAnkle, leftShoulder, leftElbow
        ]
        for j in self._jointIndicesAll:
            # print("joint : ", j) #1~13까지, 몇가지 빼고
            # self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
            self._pybullet_client.setJointMotorControl2(self._sim_model, #bodyUniqueID
                                                        j, #jointIndex
                                                        self._pybullet_client.POSITION_CONTROL, #ControlMode 
                                                        # you specify a target position for the joint, and the motor control will attempt to move the joint to that target position  
                                                        # It will apply ******(forces or torques as needed)매우중요! to achieve the desired joint position. 
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=jointFrictionForce) #(Myi) jointFrictionForce set to 0 in the above
            self._pybullet_client.setJointMotorControlMultiDof(
                self._sim_model,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, jointFrictionForce]
            )
            self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
            # if self._arg_parser is not None:
            #     fall_contact_bodies = self._arg_parser.parse_ints("fall_contact_bodies")
            self._fall_contact_body_parts = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14] #(*) hard-coded

            self._totalDofs = 7
            for dof in self._jointDofCounts:
                self._totalDofs += dof
            self.setSimTime(0) #<??tag> didnt have a look at yet.

            self._useComReward = useComReward #False?
            self.resetPose()

    def computePose(self, frameFraction):
        frameData = self._mocap_data._motion_data['Frames'][self._frame]
        frameDataNext = self._mocap_data._motion_data['Frames'][self._frameNext]

        self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pybullet_client)
        #print("self._poseInterpolator.Slerp(", frameFraction,")=", pose)
        self.computeCycleOffset()
        oldPos = self._poseInterpolator._basePos
        self._poseInterpolator._basePos = [
            oldPos[0] + self._cycleCount * self._cycleOffset[0],
            oldPos[1] + self._cycleCount * self._cycleOffset[1],
            oldPos[2] + self._cycleCount * self._cycleOffset[2]
        ]
        pose = self._poseInterpolator.GetPose()

        return pose

    def computeCycleOffset(self):
        firstFrame = 0
        lastFrame = self._mocap_data.NumFrames() - 1
        frameData = self._mocap_data._motion_data['Frames'][0]
        frameDataNext = self._mocap_data._motion_data['Frames'][lastFrame]

        basePosStart = [frameData[1], frameData[2], frameData[3]]
        basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._cycleOffset = [
            basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
            basePosEnd[2] - basePosStart[2]
        ]
        return self._cycleOffset

    def resetPose(self):
       
        pose = self.computePose(self._frameFraction) #<??tag> -> it seems like this is related to self.setSimTime(0) in init method.
        self.initializePose(self._poseInterpolator, self._sim_model, initBase=True)
    
    def initializePose(self, pose, phys_model, initBase, initializeVelocity=True):
    
        useArray = True
        if initializeVelocity:
            if initBase:
                print("Sim model comes here..") #yup
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                              pose._baseOrn)
                self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)
            if useArray:
                print("All model comes here..") #yup
                # consistent? right ?(Myi) chest = 1, neck = 2, rightHip = 3, rightKnee = 4, 
                # rightAnkle = 5, rightShoulder = 6, rightElbow = 7, leftHip = 9, 
                # leftKnee = 10, leftAnkle = 11, leftShoulder = 12,  leftElbow = 13
                indices = [chest,neck,rightHip,rightKnee, 
                        rightAnkle, rightShoulder, rightElbow,leftHip,
                        leftKnee, leftAnkle, leftShoulder,leftElbow]
                jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                                pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                                pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]
                
                jointVelocities = [pose._chestVel, pose._neckVel, pose._rightHipVel, pose._rightKneeVel,
                                pose._rightAnkleVel, pose._rightShoulderVel, pose._rightElbowVel, pose._leftHipVel,
                                pose._leftKneeVel, pose._leftAnkleVel, pose._leftShoulderVel, pose._leftElbowVel]
                self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,
                                                            jointPositions, jointVelocities)
            else:
                # print("No model goes here?") #yup
                self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot,
                                                            pose._chestVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, pose._neckVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                            pose._rightHipVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot,
                                                            pose._rightKneeVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                            pose._rightAnkleVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            pose._rightShoulderRot, pose._rightShoulderVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                            pose._rightElbowVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                            pose._leftHipVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot,
                                                            pose._leftKneeVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                            pose._leftAnkleVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            pose._leftShoulderRot, pose._leftShoulderVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot,
                                                            pose._leftElbowVel)
        else:
      
            if initBase:
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                              pose._baseOrn)
            if useArray:
                indices = [chest,neck,rightHip,rightKnee,
                        rightAnkle, rightShoulder, rightElbow,leftHip,
                        leftKnee, leftAnkle, leftShoulder,leftElbow]
                jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                                pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                                pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]
                self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,jointPositions)
                
            else:
                self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                            [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot, [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                            [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            pose._rightShoulderRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                            [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                            [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot, [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                            [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            pose._leftShoulderRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot, [0])

    def computeAndApplyPDForces(self, desiredPositions, maxForces):
        dofIndex = 7
        scaling = 1
        indices = []
        forces = []
        targetPositions=[]
        targetVelocities=[]
        kps = []
        kds = []
        
        for index in range(len(self._jointIndicesAll)):
            jointIndex = self._jointIndicesAll[index]
            indices.append(jointIndex)
            kps.append(self._kpOrg[dofIndex])
            kds.append(self._kdOrg[dofIndex])
            if self._jointDofCounts[index] == 4:
                force = [
                    scaling * maxForces[dofIndex + 0],
                    scaling * maxForces[dofIndex + 1],
                    scaling * maxForces[dofIndex + 2]
                ]
                targetVelocity = [0,0,0]
                targetPosition = [
                    desiredPositions[dofIndex + 0],
                    desiredPositions[dofIndex + 1],
                    desiredPositions[dofIndex + 2],
                    desiredPositions[dofIndex + 3]
                ]
            if self._jointDofCounts[index] == 1:
                force = [scaling * maxForces[dofIndex]]
                targetPosition = [desiredPositions[dofIndex+0]]
                targetVelocity = [0]
            forces.append(force)
            targetPositions.append(targetPosition)
            targetVelocities.append(targetVelocity)
            dofIndex += self._jointDofCounts[index]
        
            #static char* kwlist[] = { "bodyUniqueId", 
            #"jointIndices", 
            #"controlMode", "targetPositions", "targetVelocities", "forces", "positionGains", "velocityGains", "maxVelocities", "physicsClientId", NULL };
            # /// ref: https://faculty.cc.gatech.edu/~turk/my_papers/stable_pd.pdf
            self._pybullet_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                                indices,
                                                                self._pybullet_client.STABLE_PD_CONTROL,
                                                                targetPositions = targetPositions,
                                                                targetVelocities = targetVelocities,
                                                                forces=forces,
                                                                positionGains = kps,
                                                                velocityGains = kds,
                                                                )
    
    def convertActionToPose(self, action):
        pose = self._poseInterpolator.ConvertFromAction(self._pybullet_client, action)
        return pose
    
    def buildHeadingTrans(self, rootOrn):
        #align root transform 'forward' with world-space x axis
        eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
        refDir = [1, 0, 0]
        rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
        heading = math.atan2(-rotVec[2], rotVec[0])
        heading2 = eul[1]
        #print("heading=",heading)
        headingOrn = self._pybullet_client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
        return headingOrn

    def buildOriginTrans(self):
        rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)

        #print("rootPos=",rootPos, " rootOrn=",rootOrn)
        invRootPos = [-rootPos[0], 0, -rootPos[2]]
        #invOrigTransPos, invOrigTransOrn = self._pybullet_client.invertTransform(rootPos,rootOrn)
        headingOrn = self.buildHeadingTrans(rootOrn)
        #print("headingOrn=",headingOrn)
        headingMat = self._pybullet_client.getMatrixFromQuaternion(headingOrn)
        #print("headingMat=",headingMat)
        #dummy, rootOrnWithoutHeading = self._pybullet_client.multiplyTransforms([0,0,0],headingOrn, [0,0,0], rootOrn)
        #dummy, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0,0,0],rootOrnWithoutHeading, invOrigTransPos, invOrigTransOrn)

        invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                    headingOrn,
                                                                                    invRootPos,
                                                                                    [0, 0, 0, 1])
        #print("invOrigTransPos=",invOrigTransPos)
        #print("invOrigTransOrn=",invOrigTransOrn)
        invOrigTransMat = self._pybullet_client.getMatrixFromQuaternion(invOrigTransOrn)
        #print("invOrigTransMat =",invOrigTransMat )
        return invOrigTransPos, invOrigTransOrn
    
    def getPhase(self):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        phase = self._simTime / cycleTime
        phase = math.fmod(phase, 1.0)
        if (phase < 0):
            phase += 1
        return phase
    

    def getState(self):

        stateVector = []
        phase = self.getPhase()
        #print("phase=",phase)
        stateVector.append(phase)

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)

        rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                    basePos, [0, 0, 0, 1])
        #print("!!!rootPosRel =",rootPosRel )
        #print("rootTransPos=",rootTransPos)
        #print("basePos=",basePos)
        localPos, localOrn = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                    basePos, baseOrn)

        localPos = [
            localPos[0] - rootPosRel[0], localPos[1] - rootPosRel[1], localPos[2] - rootPosRel[2]
        ]
        #print("localPos=",localPos)

        stateVector.append(rootPosRel[1])

        #self.pb2dmJoints=[0,1,2,9,10,11,3,4,5,12,13,14,6,7,8]
        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        linkIndicesSim = []
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])
        
        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, linkIndicesSim, computeForwardKinematics=True, computeLinkVelocity=True)
        
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            j = self.pb2dmJoints[pbJoint]
            #print("joint order:",j)
            #ls = self._pybullet_client.getLinkState(self._sim_model, j, computeForwardKinematics=True)
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]
            linkPosLocal, linkOrnLocal = self._pybullet_client.multiplyTransforms(
                rootTransPos, rootTransOrn, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
                linkPosLocal[2] - rootPosRel[2]
            ]
            for l in linkPosLocal:
                stateVector.append(l)
            #re-order the quaternion, DeepMimic uses w,x,y,z

            if (linkOrnLocal[3] < 0):
                linkOrnLocal[0] *= -1
                linkOrnLocal[1] *= -1
                linkOrnLocal[2] *= -1
                linkOrnLocal[3] *= -1

            stateVector.append(linkOrnLocal[3])
            stateVector.append(linkOrnLocal[0])
            stateVector.append(linkOrnLocal[1])
            stateVector.append(linkOrnLocal[2])

       
        for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
            j = self.pb2dmJoints[pbJoint]
            #ls = self._pybullet_client.getLinkState(self._sim_model, j, computeLinkVelocity=True)
            ls = linkStatesSim[pbJoint]
      
            linkLinVel = ls[6]
            linkAngVel = ls[7]
            linkLinVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                                linkLinVel, [0, 0, 0, 1])
            #linkLinVelLocal=[linkLinVelLocal[0]-rootPosRel[0],linkLinVelLocal[1]-rootPosRel[1],linkLinVelLocal[2]-rootPosRel[2]]
            linkAngVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                                linkAngVel, [0, 0, 0, 1])

            for l in linkLinVelLocal:
                stateVector.append(l)
            for l in linkAngVelLocal:
                stateVector.append(l)

        #print("stateVector len=",len(stateVector))
        #for st in range (len(stateVector)):
        #  print("state[",st,"]=",stateVector[st])
        return stateVector
    
    def terminates(self):
        #check if any non-allowed body part hits the ground
        terminates = False
        pts = self._pybullet_client.getContactPoints()
        for p in pts:
            part = -1
            #ignore self-collision
            if (p[1] == p[2]):
                continue
            if (p[1] == self._sim_model):
                part = p[3]
            if (p[2] == self._sim_model):
                part = p[4]
            if (part >= 0 and part in self._fall_contact_body_parts):
                #print("terminating part:", part)
                terminates = True

        return terminates
    
    def getReward(self, pose):
        return 0
    
    
    def getCycleTime(self):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        return cycleTime
    
    def calcCycleCount(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count = math.floor(phases)
        loop = True
        #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
        return count
    
    def setSimTime(self, t):
        self._simTime = t
        # print("SetTimeTime time =",t)
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        # print("(Myi) keyFrameDuration: ", keyFrameDuration)
        cycleTime = self.getCycleTime()
        # print("(Myi) cycleTime: ", cycleTime)
        # print("self._motion_data.NumFrames()=",self._mocap_data.NumFrames())
        self._cycleCount = self.calcCycleCount(t, cycleTime)
        # print("cycles=",cycles)
        frameTime = t - self._cycleCount * cycleTime
        if (frameTime < 0):
            frameTime += cycleTime

        # print("keyFrameDuration=",keyFrameDuration)
        # print("frameTime=",frameTime)
        self._frame = int(frameTime / keyFrameDuration)
        # print("self._frame=",self._frame)

        self._frameNext = self._frame + 1
        if (self._frameNext >= self._mocap_data.NumFrames()):
            self._frameNext = self._frame

        self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)
  












